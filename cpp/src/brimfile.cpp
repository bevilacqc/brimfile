/**
 * @file brimfile.cpp
 * @brief Implementation of the C++ wrapper for the brimfile Python package.
 *
 * All Python objects are managed with the detail::PyObj RAII guard defined in
 * brimfile.hpp.  Every public function that crosses the Python/C++ boundary
 * follows the pattern:
 *
 *   1. Check that the Python interpreter is initialised.
 *   2. Build Python arguments (if any).
 *   3. Call the Python method / function.
 *   4. Check for Python exceptions and convert them to PythonError.
 *   5. Convert the Python result to the C++ return type.
 */

#include <brimfile/brimfile.hpp>

#include <cassert>
#include <cstring>
#include <sstream>

namespace brimfile {

// ============================================================================
// PythonRuntime state
// ============================================================================

bool PythonRuntime::initialized_ = false;
bool PythonRuntime::we_own_interpreter_ = false;

// ============================================================================
// Internal helpers (anonymous namespace)
// ============================================================================

namespace {

// --------------------------------------------------------------------------
// Error handling
// --------------------------------------------------------------------------

/// Fetch and clear the current Python exception; convert it to PythonError.
PythonError fetch_python_error(const std::string& context = "") {
    PyObject* ptype = nullptr;
    PyObject* pvalue = nullptr;
    PyObject* ptb = nullptr;
    PyErr_Fetch(&ptype, &pvalue, &ptb);
    PyErr_NormalizeException(&ptype, &pvalue, &ptb);

    std::string msg;
    if (pvalue) {
        PyObject* str_val = PyObject_Str(pvalue);
        if (str_val) {
            const char* s = PyUnicode_AsUTF8(str_val);
            if (s) msg = s;
            Py_DECREF(str_val);
        }
    }
    if (msg.empty()) msg = "Unknown Python error";
    if (!context.empty()) msg = context + ": " + msg;

    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptb);
    return PythonError(msg);
}

/// Throw a PythonError if a Python exception is set, using `context` as prefix.
inline void check_error(const std::string& context = "") {
    if (PyErr_Occurred()) throw fetch_python_error(context);
}

/// Throw if `obj` is nullptr (indicates a Python exception was raised).
inline PyObject* require(PyObject* obj, const std::string& context = "") {
    if (!obj) throw fetch_python_error(context);
    return obj;
}

// --------------------------------------------------------------------------
// String conversion
// --------------------------------------------------------------------------

std::string py_to_string(PyObject* obj) {
    if (!obj || obj == Py_None) return "";
    if (PyUnicode_Check(obj)) {
        const char* s = PyUnicode_AsUTF8(obj);
        if (!s) throw fetch_python_error("py_to_string: UTF-8 decode");
        return s;
    }
    // fall back: call str() on the object
    detail::PyObj s(require(PyObject_Str(obj), "py_to_string: str()"));
    const char* cs = PyUnicode_AsUTF8(s.get());
    if (!cs) throw fetch_python_error("py_to_string: UTF-8 decode");
    return cs;
}

// --------------------------------------------------------------------------
// NDArray conversion
// --------------------------------------------------------------------------

/// Convert a Python object (NumPy array or sequence) to NDArray.
NDArray py_to_ndarray(PyObject* obj) {
    if (!obj || obj == Py_None) return NDArray{};

    // Use the buffer protocol to get a direct view of the array data.
    // This works for any object that supports the buffer protocol
    // (NumPy arrays, memoryviews, bytes, etc.).
    Py_buffer view{};

    // Request a C-contiguous buffer of any format.
    if (PyObject_GetBuffer(obj, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        // The format string may include a byte-order character ('<', '>', '=', '!',
        // '@', '^') before the type character.  Skip it if present so that both
        // native "d" and little-endian "<d" (and ">d", etc.) are recognised.
        const char* fmt = view.format ? view.format : "";
        if (*fmt == '<' || *fmt == '>' || *fmt == '=' ||
            *fmt == '!' || *fmt == '@' || *fmt == '^') {
            ++fmt;
        }
        bool is_double = (fmt[0] == 'd' && fmt[1] == '\0');
        bool is_float  = (fmt[0] == 'f' && fmt[1] == '\0');

        NDArray arr;
        arr.shape.resize(static_cast<std::size_t>(view.ndim));
        for (int i = 0; i < view.ndim; ++i) {
            arr.shape[static_cast<std::size_t>(i)] =
                static_cast<std::size_t>(view.shape[i]);
        }
        std::size_t n = static_cast<std::size_t>(view.len / view.itemsize);
        arr.data.resize(n);

        if (is_double) {
            std::memcpy(arr.data.data(),
                        static_cast<const char*>(view.buf),
                        n * sizeof(double));
        } else if (is_float) {
            const float* src = static_cast<const float*>(view.buf);
            for (std::size_t i = 0; i < n; ++i) {
                arr.data[i] = static_cast<double>(src[i]);
            }
        } else {
            // For other numeric types, release the buffer and ask Python to
            // convert to float64, then retry.
            PyBuffer_Release(&view);
            detail::PyObj np(require(
                PyImport_ImportModule("numpy"), "py_to_ndarray: import numpy"));
            detail::PyObj float64(
                require(PyObject_GetAttrString(np.get(), "float64"),
                        "numpy.float64"));
            detail::PyObj astype_name(
                require(PyUnicode_FromString("astype"), "astype name"));
            detail::PyObj converted(
                require(PyObject_CallMethodObjArgs(
                            obj, astype_name.get(), float64.get(), nullptr),
                        "astype(float64)"));
            return py_to_ndarray(converted.get());
        }

        PyBuffer_Release(&view);
        return arr;
    }

    // Buffer protocol not supported or array is non-contiguous; force a
    // C-contiguous float64 copy via numpy.ascontiguousarray().
    PyErr_Clear();
    detail::PyObj np(require(
        PyImport_ImportModule("numpy"), "py_to_ndarray: import numpy"));
    detail::PyObj float64(
        require(PyObject_GetAttrString(np.get(), "float64"), "numpy.float64"));
    detail::PyObj ascontig_name(
        require(PyUnicode_FromString("ascontiguousarray"), "ascontiguousarray name"));
    // numpy.ascontiguousarray(obj, dtype=float64) always returns a writable,
    // C-contiguous copy with the requested dtype.
    detail::PyObj arr_obj(
        require(PyObject_CallMethodObjArgs(
                    np.get(), ascontig_name.get(), obj, float64.get(), nullptr),
                "numpy.ascontiguousarray"));
    // Now retry with the guaranteed-contiguous array.
    Py_buffer view2{};
    if (PyObject_GetBuffer(arr_obj.get(), &view2, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0)
        throw fetch_python_error("py_to_ndarray: buffer on contiguous array");
    const char* fmt2 = view2.format ? view2.format : "";
    if (*fmt2 == '<' || *fmt2 == '>' || *fmt2 == '=' ||
        *fmt2 == '!' || *fmt2 == '@' || *fmt2 == '^') ++fmt2;
    NDArray arr;
    arr.shape.resize(static_cast<std::size_t>(view2.ndim));
    for (int i = 0; i < view2.ndim; ++i)
        arr.shape[static_cast<std::size_t>(i)] =
            static_cast<std::size_t>(view2.shape[i]);
    std::size_t n2 = static_cast<std::size_t>(view2.len / view2.itemsize);
    arr.data.resize(n2);
    if (fmt2[0] == 'd' && fmt2[1] == '\0') {
        std::memcpy(arr.data.data(), static_cast<const char*>(view2.buf),
                    n2 * sizeof(double));
    } else if (fmt2[0] == 'f' && fmt2[1] == '\0') {
        const float* src = static_cast<const float*>(view2.buf);
        for (std::size_t i = 0; i < n2; ++i)
            arr.data[i] = static_cast<double>(src[i]);
    } else {
        PyBuffer_Release(&view2);
        throw PythonError(std::string("py_to_ndarray: unsupported dtype '") + fmt2 + "'");
    }
    PyBuffer_Release(&view2);
    return arr;
}

/// Convert an NDArray to a NumPy ndarray object.
detail::PyObj ndarray_to_py(const NDArray& arr) {
    detail::PyObj np(
        require(PyImport_ImportModule("numpy"), "ndarray_to_py: import numpy"));

    if (arr.data.empty()) {
        // Return an empty 1-D numpy array.
        detail::PyObj zeros_name(
            require(PyUnicode_FromString("zeros"), "zeros name"));
        detail::PyObj size(require(PyLong_FromLong(0), "size=0"));
        detail::PyObj float64(
            require(PyObject_GetAttrString(np.get(), "float64"), "numpy.float64"));
        detail::PyObj result(require(
            PyObject_CallMethodObjArgs(
                np.get(), zeros_name.get(), size.get(), float64.get(), nullptr),
            "numpy.zeros"));
        return result;
    }

    // Build a bytes object from the raw double data.
    detail::PyObj bytes_obj(require(
        PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(arr.data.data()),
            static_cast<Py_ssize_t>(arr.data.size() * sizeof(double))),
        "PyBytes_FromStringAndSize"));

    // numpy.frombuffer(bytes_obj, dtype=float64)
    detail::PyObj float64(
        require(PyObject_GetAttrString(np.get(), "float64"), "numpy.float64"));
    detail::PyObj frombuffer_name(
        require(PyUnicode_FromString("frombuffer"), "frombuffer name"));
    detail::PyObj flat(require(
        PyObject_CallMethodObjArgs(
            np.get(), frombuffer_name.get(),
            bytes_obj.get(), float64.get(), nullptr),
        "numpy.frombuffer"));

    // Reshape to the desired shape.
    detail::PyObj shape_tuple(PyTuple_New(
        static_cast<Py_ssize_t>(arr.shape.size())));
    require(shape_tuple.get(), "PyTuple_New for shape");
    for (std::size_t i = 0; i < arr.shape.size(); ++i) {
        PyTuple_SET_ITEM(
            shape_tuple.get(),
            static_cast<Py_ssize_t>(i),
            PyLong_FromSsize_t(static_cast<Py_ssize_t>(arr.shape[i])));
    }
    detail::PyObj reshape_name(
        require(PyUnicode_FromString("reshape"), "reshape name"));
    detail::PyObj result(require(
        PyObject_CallMethodObjArgs(
            flat.get(), reshape_name.get(), shape_tuple.get(), nullptr),
        "ndarray.reshape"));
    return result;
}

// --------------------------------------------------------------------------
// MetadataItem conversion
// --------------------------------------------------------------------------

MetadataItem py_to_metadata_item(PyObject* obj) {
    MetadataItem item;
    if (!obj || obj == Py_None) return item;

    // The Python MetadataItem has .value and .units attributes.
    detail::PyObj py_val(PyObject_GetAttrString(obj, "value"));
    detail::PyObj py_units(PyObject_GetAttrString(obj, "units"));
    PyErr_Clear(); // non-fatal if attributes missing

    if (py_val && py_val.get() != Py_None) {
        item.value = py_to_string(py_val.get());
        item.has_value = true;
    }
    if (py_units && py_units.get() != Py_None) {
        item.units = py_to_string(py_units.get());
    }
    return item;
}

/// Convert a Python dict of {str: MetadataItem} to a C++ map.
std::map<std::string, MetadataItem>
py_dict_to_metadata_map(PyObject* dict) {
    std::map<std::string, MetadataItem> result;
    if (!dict || dict == Py_None) return result;
    if (!PyDict_Check(dict)) return result;

    PyObject* key = nullptr;
    PyObject* value = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict, &pos, &key, &value)) {
        std::string k = py_to_string(key);
        result[k] = py_to_metadata_item(value);
    }
    return result;
}

// --------------------------------------------------------------------------
// GroupInfo conversion
// --------------------------------------------------------------------------

GroupInfo py_dict_to_group_info(PyObject* dict) {
    GroupInfo info;
    if (!dict || !PyDict_Check(dict)) return info;

    // PyDict_GetItemString returns borrowed references – do NOT wrap in PyObj.
    PyObject* py_index  = PyDict_GetItemString(dict, "index");
    PyObject* py_name   = PyDict_GetItemString(dict, "name");
    PyObject* py_custom = PyDict_GetItemString(dict, "custom_name");

    if (py_index && py_index != Py_None)
        info.index = static_cast<int>(PyLong_AsLong(py_index));
    if (py_name && py_name != Py_None)
        info.name = py_to_string(py_name);
    if (py_custom && py_custom != Py_None)
        info.custom_name = py_to_string(py_custom);
    return info;
}

std::vector<GroupInfo> py_list_to_group_info_vec(PyObject* lst) {
    std::vector<GroupInfo> result;
    if (!lst || lst == Py_None) return result;
    Py_ssize_t n = PyList_Size(lst);
    for (Py_ssize_t i = 0; i < n; ++i) {
        result.push_back(py_dict_to_group_info(PyList_GET_ITEM(lst, i)));
    }
    return result;
}

// --------------------------------------------------------------------------
// SpectrumResult conversion
// --------------------------------------------------------------------------

SpectrumResult py_tuple_to_spectrum_result(PyObject* tpl) {
    SpectrumResult sr;
    if (!tpl) throw fetch_python_error("py_tuple_to_spectrum_result");
    Py_ssize_t sz = PyTuple_Size(tpl);
    if (sz < 2)
        throw PythonError("Expected tuple of length >= 2 for spectrum result");

    sr.psd       = py_to_ndarray(PyTuple_GET_ITEM(tpl, 0));
    sr.frequency = py_to_ndarray(PyTuple_GET_ITEM(tpl, 1));
    if (sz >= 3) {
        PyObject* u = PyTuple_GET_ITEM(tpl, 2);
        if (u && u != Py_None) sr.psd_units = py_to_string(u);
    }
    if (sz >= 4) {
        PyObject* u = PyTuple_GET_ITEM(tpl, 3);
        if (u && u != Py_None) sr.freq_units = py_to_string(u);
    }
    return sr;
}

// --------------------------------------------------------------------------
// ImageResult conversion
// --------------------------------------------------------------------------

std::array<MetadataItem, 3> py_tuple_to_px_size(PyObject* tpl) {
    std::array<MetadataItem, 3> px{};
    if (!tpl || tpl == Py_None) return px;
    Py_ssize_t sz = PyTuple_Size(tpl);
    for (Py_ssize_t i = 0; i < sz && i < 3; ++i) {
        px[static_cast<std::size_t>(i)] =
            py_to_metadata_item(PyTuple_GET_ITEM(tpl, i));
    }
    return px;
}

ImageResult py_tuple_to_image_result(PyObject* tpl) {
    ImageResult ir;
    if (!tpl) throw fetch_python_error("py_tuple_to_image_result");
    Py_ssize_t sz = PyTuple_Size(tpl);
    if (sz < 2)
        throw PythonError("Expected tuple of length >= 2 for image result");

    ir.image   = py_to_ndarray(PyTuple_GET_ITEM(tpl, 0));
    ir.px_size = py_tuple_to_px_size(PyTuple_GET_ITEM(tpl, 1));
    return ir;
}

// --------------------------------------------------------------------------
// Quantity / PeakType enum → Python enum value
// --------------------------------------------------------------------------

detail::PyObj get_py_quantity(PyObject* ar_class, Quantity qt) {
    // Get AnalysisResults.Quantity enum and call its value.
    detail::PyObj qty_enum(require(
        PyObject_GetAttrString(ar_class, "Quantity"), "AnalysisResults.Quantity"));

    const char* name = nullptr;
    switch (qt) {
    case Quantity::Shift:             name = "Shift";            break;
    case Quantity::Width:             name = "Width";            break;
    case Quantity::Amplitude:         name = "Amplitude";        break;
    case Quantity::Offset:            name = "Offset";           break;
    case Quantity::R2:                name = "R2";               break;
    case Quantity::RMSE:              name = "RMSE";             break;
    case Quantity::Cov_matrix:        name = "Cov_matrix";       break;
    case Quantity::Elastic_contrast:  name = "Elastic_contrast"; break;
    case Quantity::Viscous_contrast:  name = "Viscous_contrast"; break;
    }
    return detail::PyObj(require(
        PyObject_GetAttrString(qty_enum.get(), name), name));
}

detail::PyObj get_py_peak_type(PyObject* ar_class, PeakType pt) {
    detail::PyObj pt_enum(require(
        PyObject_GetAttrString(ar_class, "PeakType"), "AnalysisResults.PeakType"));

    const char* name = nullptr;
    switch (pt) {
    case PeakType::AntiStokes: name = "AntiStokes"; break;
    case PeakType::Stokes:     name = "Stokes";     break;
    case PeakType::Average:    name = "average";    break;
    }
    return detail::PyObj(require(
        PyObject_GetAttrString(pt_enum.get(), name), name));
}

// --------------------------------------------------------------------------
// BrimMetadata::Type → Python string
// --------------------------------------------------------------------------

const char* metadata_type_to_str(BrimMetadata::Type type) {
    switch (type) {
    case BrimMetadata::Type::Experiment:  return "Experiment";
    case BrimMetadata::Type::Optics:      return "Optics";
    case BrimMetadata::Type::Brillouin:   return "Brillouin";
    case BrimMetadata::Type::Acquisition: return "Acquisition";
    case BrimMetadata::Type::Spectrometer:return "Spectrometer";
    }
    return "Experiment";
}

// Helper: get brimfile.Metadata.Type enum value for the given type.
detail::PyObj get_py_metadata_type(PyObject* metadata_obj,
                                   BrimMetadata::Type type) {
    // metadata_obj is a brimfile.Metadata instance;
    // get its class to access the Type enum.
    detail::PyObj cls(require(PyObject_GetAttrString(metadata_obj, "__class__"),
                              "Metadata.__class__"));
    detail::PyObj type_enum(
        require(PyObject_GetAttrString(cls.get(), "Type"), "Metadata.Type"));
    return detail::PyObj(require(
        PyObject_GetAttrString(type_enum.get(),
                               metadata_type_to_str(type)),
        metadata_type_to_str(type)));
}

// --------------------------------------------------------------------------
// AnalysisData → Python dict
// --------------------------------------------------------------------------

detail::PyObj analysis_data_to_py_dict(const AnalysisData& ad) {
    detail::PyObj d(require(PyDict_New(), "PyDict_New"));

    auto add_array = [&](const NDArray& arr,
                         const char* key,
                         const std::string& units_key,
                         const std::string& units_val) {
        if (arr.empty()) return;
        detail::PyObj py_arr = ndarray_to_py(arr);
        detail::PyObj k(require(PyUnicode_FromString(key), key));
        if (PyDict_SetItem(d.get(), k.get(), py_arr.get()) < 0)
            throw fetch_python_error("PyDict_SetItem");
        if (!units_val.empty()) {
            detail::PyObj uk(require(PyUnicode_FromString(units_key.c_str()),
                                     units_key.c_str()));
            detail::PyObj uv(require(PyUnicode_FromString(units_val.c_str()),
                                     units_val.c_str()));
            if (PyDict_SetItem(d.get(), uk.get(), uv.get()) < 0)
                throw fetch_python_error("PyDict_SetItem units");
        }
    };

    add_array(ad.shift,     "shift",     "shift_units",     ad.shift_units);
    add_array(ad.width,     "width",     "width_units",     ad.width_units);
    add_array(ad.amplitude, "amplitude", "amplitude_units", ad.amplitude_units);
    add_array(ad.offset,    "offset",    "offset_units",    ad.offset_units);
    add_array(ad.R2,        "R2",        "R2_units",        "");
    add_array(ad.RMSE,      "RMSE",      "RMSE_units",      "");

    return d;
}

} // anonymous namespace

// ============================================================================
// PythonRuntime
// ============================================================================

void PythonRuntime::init() {
    if (initialized_) return;

    if (!Py_IsInitialized()) {
        Py_Initialize();
        we_own_interpreter_ = true;
    }

    // Verify that the brimfile package is importable.
    detail::PyObj bf(PyImport_ImportModule("brimfile"));
    if (!bf) throw fetch_python_error("PythonRuntime::init: import brimfile");

    // Verify that numpy is importable (required for array conversion).
    detail::PyObj np(PyImport_ImportModule("numpy"));
    if (!np) throw fetch_python_error("PythonRuntime::init: import numpy");

    initialized_ = true;
}

// ============================================================================
// BrimFile
// ============================================================================

BrimFile::BrimFile(const std::string& filename, const std::string& mode) {
    if (!PythonRuntime::is_initialized())
        PythonRuntime::init();

    detail::PyObj brim(require(
        PyImport_ImportModule("brimfile"), "BrimFile: import brimfile"));
    detail::PyObj file_cls(require(
        PyObject_GetAttrString(brim.get(), "File"), "BrimFile: File class"));

    detail::PyObj py_fname(
        require(PyUnicode_FromString(filename.c_str()), "filename"));
    detail::PyObj py_mode(
        require(PyUnicode_FromString(mode.c_str()), "mode"));
    detail::PyObj args(require(PyTuple_Pack(2, py_fname.get(), py_mode.get()),
                               "PyTuple_Pack"));
    py_obj_ = detail::PyObj(
        require(PyObject_Call(file_cls.get(), args.get(), nullptr),
                "File()"));
}

BrimFile BrimFile::create(const std::string& filename) {
    if (!PythonRuntime::is_initialized())
        PythonRuntime::init();

    detail::PyObj brim(require(
        PyImport_ImportModule("brimfile"), "BrimFile::create: import brimfile"));
    detail::PyObj file_cls(require(
        PyObject_GetAttrString(brim.get(), "File"),
        "BrimFile::create: File class"));
    detail::PyObj create_meth(require(
        PyObject_GetAttrString(file_cls.get(), "create"),
        "BrimFile::create: File.create"));

    detail::PyObj py_fname(
        require(PyUnicode_FromString(filename.c_str()), "filename"));
    detail::PyObj args(require(PyTuple_Pack(1, py_fname.get()), "args"));
    detail::PyObj py_file(
        require(PyObject_Call(create_meth.get(), args.get(), nullptr),
                "File.create()"));

    return BrimFile(std::move(py_file));
}

BrimFile::~BrimFile() {
    if (py_obj_) {
        try { close(); } catch (...) {}
    }
}

BrimFile::BrimFile(BrimFile&& o) noexcept : py_obj_(std::move(o.py_obj_)) {}

BrimFile& BrimFile::operator=(BrimFile&& o) noexcept {
    if (this != &o) {
        if (py_obj_) {
            try { close(); } catch (...) {}
        }
        py_obj_ = std::move(o.py_obj_);
    }
    return *this;
}

void BrimFile::close() {
    if (!py_obj_) return;
    detail::PyObj result(
        PyObject_CallMethod(py_obj_.get(), "close", nullptr));
    check_error("BrimFile::close");
    py_obj_ = detail::PyObj{}; // release the Python object
}

bool BrimFile::is_read_only() const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "is_read_only", nullptr),
        "BrimFile::is_read_only"));
    return PyObject_IsTrue(result.get()) != 0;
}

std::string BrimFile::filename() const {
    detail::PyObj prop(require(
        PyObject_GetAttrString(py_obj_.get(), "filename"),
        "BrimFile::filename"));
    return py_to_string(prop.get());
}

std::vector<GroupInfo> BrimFile::list_data_groups(
    bool retrieve_custom_name) const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "list_data_groups", "O",
                            retrieve_custom_name ? Py_True : Py_False),
        "BrimFile::list_data_groups"));
    return py_list_to_group_info_vec(result.get());
}

BrimData BrimFile::get_data(int index) const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "get_data", "i", index),
        "BrimFile::get_data"));
    return BrimData(std::move(result));
}

BrimData BrimFile::create_data_group(const NDArray& psd,
                                     const NDArray& frequency,
                                     const std::array<double, 3>& px_size_um,
                                     const std::string& name) const {
    detail::PyObj py_psd = ndarray_to_py(psd);
    detail::PyObj py_freq = ndarray_to_py(frequency);

    // Build px_size_um as a Python tuple of 3 elements.
    detail::PyObj py_px(require(PyTuple_New(3), "px_size tuple"));
    for (int i = 0; i < 3; ++i) {
        PyTuple_SET_ITEM(py_px.get(), i,
                         PyFloat_FromDouble(px_size_um[static_cast<std::size_t>(i)]));
    }

    // Build keyword arguments.
    detail::PyObj kwargs(require(PyDict_New(), "kwargs dict"));
    if (!name.empty()) {
        detail::PyObj k(require(PyUnicode_FromString("name"), "name key"));
        detail::PyObj v(require(PyUnicode_FromString(name.c_str()), "name val"));
        if (PyDict_SetItem(kwargs.get(), k.get(), v.get()) < 0)
            throw fetch_python_error("create_data_group: set name");
    }

    detail::PyObj args(require(
        PyTuple_Pack(3, py_psd.get(), py_freq.get(), py_px.get()), "args"));
    detail::PyObj result(require(
        PyObject_Call(
            require(PyObject_GetAttrString(py_obj_.get(), "create_data_group"),
                    "create_data_group method"),
            args.get(), kwargs.get()),
        "BrimFile::create_data_group"));
    return BrimData(std::move(result));
}

// ============================================================================
// BrimData
// ============================================================================

std::string BrimData::get_name() const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "get_name", nullptr),
        "BrimData::get_name"));
    return py_to_string(result.get());
}

int BrimData::get_index() const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "get_index", nullptr),
        "BrimData::get_index"));
    return static_cast<int>(PyLong_AsLong(result.get()));
}

SpectrumResult BrimData::get_PSD_as_spatial_map(bool broadcast_frequency) const {
    // Use keyword argument broadcast_frequency.
    detail::PyObj kwargs(require(PyDict_New(), "kwargs"));
    detail::PyObj k(require(PyUnicode_FromString("broadcast_frequency"), "key"));
    PyObject* py_flag = broadcast_frequency ? Py_True : Py_False;
    if (PyDict_SetItem(kwargs.get(), k.get(), py_flag) < 0)
        throw fetch_python_error("get_PSD_as_spatial_map: set kwarg");
    detail::PyObj args(require(PyTuple_New(0), "empty args"));
    detail::PyObj meth(require(
        PyObject_GetAttrString(py_obj_.get(), "get_PSD_as_spatial_map"),
        "get_PSD_as_spatial_map attr"));
    detail::PyObj result(require(
        PyObject_Call(meth.get(), args.get(), kwargs.get()),
        "BrimData::get_PSD_as_spatial_map"));
    return py_tuple_to_spectrum_result(result.get());
}

SpectrumResult BrimData::get_spectrum_in_image(
    const std::array<int, 3>& coord) const {
    // Build a Python tuple (z, y, x).
    detail::PyObj py_coord(require(PyTuple_New(3), "coord tuple"));
    for (int i = 0; i < 3; ++i) {
        PyTuple_SET_ITEM(py_coord.get(), i,
                         PyLong_FromLong(coord[static_cast<std::size_t>(i)]));
    }
    // Use "(O)" to pass the coord tuple as a single argument rather than
    // unpacking its three integers as three separate arguments.
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "get_spectrum_in_image", "(O)",
                            py_coord.get()),
        "BrimData::get_spectrum_in_image"));
    return py_tuple_to_spectrum_result(result.get());
}

BrimMetadata BrimData::get_metadata() const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "get_metadata", nullptr),
        "BrimData::get_metadata"));
    return BrimMetadata(std::move(result));
}

std::vector<GroupInfo> BrimData::list_analysis_results(
    bool retrieve_custom_name) const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "list_AnalysisResults", "O",
                            retrieve_custom_name ? Py_True : Py_False),
        "BrimData::list_analysis_results"));
    return py_list_to_group_info_vec(result.get());
}

BrimAnalysisResults BrimData::get_analysis_results(int index) const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "get_analysis_results", "i", index),
        "BrimData::get_analysis_results"));
    return BrimAnalysisResults(std::move(result));
}

BrimAnalysisResults BrimData::create_analysis_results_group(
    const AnalysisData& data_as,
    const AnalysisData& data_s,
    const std::string& name,
    const std::string& fit_model) const {

    detail::PyObj py_as = analysis_data_to_py_dict(data_as);
    bool has_stokes = !data_s.shift.empty() || !data_s.width.empty();
    detail::PyObj py_s(has_stokes ? analysis_data_to_py_dict(data_s).release()
                                  : (Py_INCREF(Py_None), Py_None));

    detail::PyObj kwargs(require(PyDict_New(), "kwargs"));
    if (!name.empty()) {
        detail::PyObj k(require(PyUnicode_FromString("name"), "name key"));
        detail::PyObj v(require(PyUnicode_FromString(name.c_str()), "name"));
        if (PyDict_SetItem(kwargs.get(), k.get(), v.get()) < 0)
            throw fetch_python_error("create_ar: set name");
    }
    if (!fit_model.empty()) {
        // brimfile.Data.AnalysisResults.FitModel.<fit_model>
        detail::PyObj brim(require(
            PyImport_ImportModule("brimfile"), "import brimfile"));
        detail::PyObj data_cls(require(
            PyObject_GetAttrString(brim.get(), "Data"), "brimfile.Data"));
        detail::PyObj ar_cls(require(
            PyObject_GetAttrString(data_cls.get(), "AnalysisResults"),
            "Data.AnalysisResults"));
        detail::PyObj fm_enum(require(
            PyObject_GetAttrString(ar_cls.get(), "FitModel"),
            "AnalysisResults.FitModel"));
        detail::PyObj fm_val(require(
            PyObject_GetAttrString(fm_enum.get(), fit_model.c_str()),
            fit_model.c_str()));
        detail::PyObj k(require(PyUnicode_FromString("fit_model"), "fit_model key"));
        if (PyDict_SetItem(kwargs.get(), k.get(), fm_val.get()) < 0)
            throw fetch_python_error("create_ar: set fit_model");
    }

    detail::PyObj args(require(
        PyTuple_Pack(2, py_as.get(), py_s.get()), "create_ar args"));
    detail::PyObj meth(require(
        PyObject_GetAttrString(py_obj_.get(), "create_analysis_results_group"),
        "create_analysis_results_group attr"));
    detail::PyObj result(require(
        PyObject_Call(meth.get(), args.get(), kwargs.get()),
        "BrimData::create_analysis_results_group"));
    return BrimAnalysisResults(std::move(result));
}

// ============================================================================
// BrimAnalysisResults
// ============================================================================

std::string BrimAnalysisResults::get_name() const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "get_name", nullptr),
        "BrimAnalysisResults::get_name"));
    return py_to_string(result.get());
}

std::optional<std::string> BrimAnalysisResults::get_units(
    Quantity qt, PeakType pt, int index) const {

    // Retrieve the class to access enum values.
    detail::PyObj cls(require(
        PyObject_GetAttrString(py_obj_.get(), "__class__"),
        "BrimAnalysisResults::get_units: __class__"));

    detail::PyObj py_qt = get_py_quantity(cls.get(), qt);
    detail::PyObj py_pt = get_py_peak_type(cls.get(), pt);
    detail::PyObj py_idx(require(PyLong_FromLong(index), "index"));

    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "get_units", "OOO",
                            py_qt.get(), py_pt.get(), py_idx.get()),
        "BrimAnalysisResults::get_units"));

    if (result.get() == Py_None) return std::nullopt;
    return py_to_string(result.get());
}

ImageResult BrimAnalysisResults::get_image(
    Quantity qt, PeakType pt, int index) const {

    detail::PyObj cls(require(
        PyObject_GetAttrString(py_obj_.get(), "__class__"),
        "BrimAnalysisResults::get_image: __class__"));

    detail::PyObj py_qt = get_py_quantity(cls.get(), qt);
    detail::PyObj py_pt = get_py_peak_type(cls.get(), pt);
    detail::PyObj py_idx(require(PyLong_FromLong(index), "index"));

    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "get_image", "OOO",
                            py_qt.get(), py_pt.get(), py_idx.get()),
        "BrimAnalysisResults::get_image"));

    return py_tuple_to_image_result(result.get());
}

NDArray BrimAnalysisResults::get_quantity_at_pixel(
    const std::array<int, 3>& coord,
    Quantity qt, PeakType pt, int index) const {

    detail::PyObj cls(require(
        PyObject_GetAttrString(py_obj_.get(), "__class__"),
        "BrimAnalysisResults::get_quantity_at_pixel: __class__"));

    detail::PyObj py_coord(require(PyTuple_New(3), "coord"));
    for (int i = 0; i < 3; ++i) {
        PyTuple_SET_ITEM(py_coord.get(), i,
                         PyLong_FromLong(coord[static_cast<std::size_t>(i)]));
    }
    detail::PyObj py_qt = get_py_quantity(cls.get(), qt);
    detail::PyObj py_pt = get_py_peak_type(cls.get(), pt);
    detail::PyObj py_idx(require(PyLong_FromLong(index), "index"));

    // Use "(O)" so the coord tuple is passed as a single argument.
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "get_quantity_at_pixel", "(OOOO)",
                            py_coord.get(), py_qt.get(),
                            py_pt.get(), py_idx.get()),
        "BrimAnalysisResults::get_quantity_at_pixel"));

    return py_to_ndarray(result.get());
}

std::vector<std::string> BrimAnalysisResults::list_existing_peak_types(
    int index) const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "list_existing_peak_types",
                            "i", index),
        "BrimAnalysisResults::list_existing_peak_types"));

    std::vector<std::string> out;
    Py_ssize_t sz = PyTuple_Size(result.get());
    for (Py_ssize_t i = 0; i < sz; ++i) {
        PyObject* item = PyTuple_GET_ITEM(result.get(), i);
        // Each item is a PeakType enum; get its .value attribute.
        detail::PyObj val(PyObject_GetAttrString(item, "value"));
        if (val) out.push_back(py_to_string(val.get()));
    }
    return out;
}

std::vector<std::string> BrimAnalysisResults::list_existing_quantities(
    PeakType pt, int index) const {

    detail::PyObj cls(require(
        PyObject_GetAttrString(py_obj_.get(), "__class__"),
        "list_existing_quantities: __class__"));

    detail::PyObj py_pt = get_py_peak_type(cls.get(), pt);
    detail::PyObj py_idx(require(PyLong_FromLong(index), "index"));

    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "list_existing_quantities",
                            "OO", py_pt.get(), py_idx.get()),
        "BrimAnalysisResults::list_existing_quantities"));

    std::vector<std::string> out;
    Py_ssize_t sz = PyTuple_Size(result.get());
    for (Py_ssize_t i = 0; i < sz; ++i) {
        PyObject* item = PyTuple_GET_ITEM(result.get(), i);
        // Each item is a Quantity enum; get its .value attribute.
        detail::PyObj val(PyObject_GetAttrString(item, "value"));
        if (val) out.push_back(py_to_string(val.get()));
    }
    return out;
}

std::string BrimAnalysisResults::fit_model() const {
    detail::PyObj prop(require(
        PyObject_GetAttrString(py_obj_.get(), "fit_model"),
        "BrimAnalysisResults::fit_model"));
    // fit_model is a property returning a FitModel enum; get its .value.
    detail::PyObj val(PyObject_GetAttrString(prop.get(), "value"));
    if (val && val.get() != Py_None) return py_to_string(val.get());
    return "";
}

// ============================================================================
// BrimMetadata
// ============================================================================

MetadataItem BrimMetadata::operator[](const std::string& key) const {
    detail::PyObj py_key(require(PyUnicode_FromString(key.c_str()), "key"));
    detail::PyObj result(require(
        PyObject_GetItem(py_obj_.get(), py_key.get()),
        "BrimMetadata::operator[]"));
    return py_to_metadata_item(result.get());
}

std::map<std::string, MetadataItem> BrimMetadata::to_dict(Type type) const {
    detail::PyObj py_type = get_py_metadata_type(py_obj_.get(), type);
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "to_dict", "O", py_type.get()),
        "BrimMetadata::to_dict"));
    return py_dict_to_metadata_map(result.get());
}

std::map<std::string, std::map<std::string, MetadataItem>>
BrimMetadata::all_to_dict() const {
    detail::PyObj result(require(
        PyObject_CallMethod(py_obj_.get(), "all_to_dict", nullptr),
        "BrimMetadata::all_to_dict"));

    std::map<std::string, std::map<std::string, MetadataItem>> out;
    if (!result || !PyDict_Check(result.get())) return out;

    PyObject* k = nullptr;
    PyObject* v = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(result.get(), &pos, &k, &v)) {
        std::string type_name = py_to_string(k);
        out[type_name] = py_dict_to_metadata_map(v);
    }
    return out;
}

void BrimMetadata::add(Type type,
                       const std::map<std::string, MetadataItem>& items,
                       bool local) {
    // Import brimfile to get the Metadata.Item class.
    detail::PyObj brim(require(
        PyImport_ImportModule("brimfile"), "BrimMetadata::add: import brimfile"));
    detail::PyObj meta_cls(require(
        PyObject_GetAttrString(brim.get(), "Metadata"), "Metadata class"));
    detail::PyObj item_cls(require(
        PyObject_GetAttrString(meta_cls.get(), "Item"), "Metadata.Item"));

    // Build a Python dict of {str: Metadata.Item}.
    detail::PyObj py_dict(require(PyDict_New(), "PyDict_New"));
    for (const auto& [attr_name, item] : items) {
        // Construct Metadata.Item(value, units).
        PyObject* py_val_raw = Py_None;
        detail::PyObj py_val_owned;
        if (item.has_value && !item.value.empty()) {
            // Only treat as a number if ALL characters were consumed.
            // std::stod("2024-01-01T...") would otherwise return 2024.0
            // silently, which would break string-typed schema fields.
            bool is_number = false;
            double num_val = 0.0;
            try {
                std::size_t pos = 0;
                num_val = std::stod(item.value, &pos);
                is_number = (pos == item.value.size());
            } catch (...) {
                is_number = false;
            }
            if (is_number) {
                py_val_owned = detail::PyObj(PyFloat_FromDouble(num_val));
            } else {
                py_val_owned = detail::PyObj(
                    require(PyUnicode_FromString(item.value.c_str()), "value str"));
            }
            py_val_raw = py_val_owned.get();
        }

        detail::PyObj py_units(item.units.empty()
                                   ? (Py_INCREF(Py_None), Py_None)
                                   : require(PyUnicode_FromString(item.units.c_str()),
                                             "units str"));

        detail::PyObj args(require(
            PyTuple_Pack(2, py_val_raw, py_units.get()), "Metadata.Item args"));
        detail::PyObj py_item(require(
            PyObject_Call(item_cls.get(), args.get(), nullptr),
            "Metadata.Item()"));

        detail::PyObj key(
            require(PyUnicode_FromString(attr_name.c_str()), "attr key"));
        if (PyDict_SetItem(py_dict.get(), key.get(), py_item.get()) < 0)
            throw fetch_python_error("BrimMetadata::add PyDict_SetItem");
    }

    detail::PyObj py_type = get_py_metadata_type(py_obj_.get(), type);

    // Call metadata.add(type, dict, local=local).
    detail::PyObj kwargs(require(PyDict_New(), "kwargs"));
    detail::PyObj k_local(require(PyUnicode_FromString("local"), "local key"));
    if (PyDict_SetItem(kwargs.get(), k_local.get(),
                       local ? Py_True : Py_False) < 0)
        throw fetch_python_error("BrimMetadata::add set local kwarg");

    detail::PyObj args(require(
        PyTuple_Pack(2, py_type.get(), py_dict.get()), "add args"));
    detail::PyObj meth(require(
        PyObject_GetAttrString(py_obj_.get(), "add"), "Metadata.add attr"));
    detail::PyObj result(require(
        PyObject_Call(meth.get(), args.get(), kwargs.get()),
        "BrimMetadata::add call"));
}

} // namespace brimfile
