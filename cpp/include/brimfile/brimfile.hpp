/**
 * @file brimfile.hpp
 * @brief C++ wrapper for the brimfile Python package.
 *
 * This header provides a C++ API for reading and writing brim files (Brillouin
 * microscopy data files) by embedding the Python brimfile package via the CPython
 * C API (https://docs.python.org/3/extending/embedding.html).
 *
 * Prerequisites:
 *   - Python 3.11+ with the brimfile package installed.
 *   - Python development headers (python3-dev / python3-devel).
 *   - NumPy installed in the Python environment.
 *
 * Usage:
 * @code
 *   #include <brimfile/brimfile.hpp>
 *
 *   brimfile::PythonRuntime::init(); // call once before using the library
 *
 *   brimfile::BrimFile f("path/to/file.brim.zarr");
 *   auto d = f.get_data(0);
 *   auto [psd, freq, psd_units, freq_units] = d.get_PSD_as_spatial_map();
 *   f.close();
 * @endcode
 *
 * Thread safety: the Python GIL must be held when calling any function in this
 * library.  If you use Python from multiple threads, acquire the GIL with
 * PyGILState_Ensure/PyGILState_Release around calls into this library.
 */

#pragma once

#include <Python.h>

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace brimfile {

// ============================================================================
// Exceptions
// ============================================================================

/// Base exception for all brimfile C++ wrapper errors.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

/// Exception representing an error thrown by the Python brimfile package.
class PythonError : public Error {
public:
    explicit PythonError(const std::string& msg) : Error(msg) {}
};

// ============================================================================
// Data structures
// ============================================================================

/**
 * @brief N-dimensional array backed by a contiguous double buffer.
 *
 * `data` holds all elements in C (row-major) order; `shape` lists the size of
 * each dimension from outermost to innermost, matching NumPy's default layout.
 */
struct NDArray {
    std::vector<double> data;
    std::vector<std::size_t> shape;

    /// Total number of elements (product of all shape dimensions).
    /// A 0-d (scalar) array has an empty shape and exactly one element.
    std::size_t size() const {
        std::size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }

    /// Returns true if the array has no elements.
    /// Note: a 0-d scalar (empty shape) has one element and is NOT considered empty.
    bool empty() const { return data.empty(); }
};

/**
 * @brief A single metadata attribute with an optional value and units string.
 *
 * The `value` field holds the string representation of the attribute. Numeric
 * values are represented as their decimal string (e.g. "22.0").  When no value
 * is stored the string is empty and `has_value` is false.
 */
struct MetadataItem {
    std::string value;  ///< String representation of the attribute value.
    std::string units;  ///< Units string, empty when not defined.
    bool has_value = false; ///< False when the field is absent/missing.
};

/**
 * @brief Descriptor for a data group or analysis-results group.
 *
 * Returned by list_data_groups() and BrimData::list_analysis_results().
 */
struct GroupInfo {
    int index = 0;            ///< Numeric index of the group.
    std::string name;         ///< Internal group name (e.g. "Brillouin_data_0").
    std::string custom_name;  ///< User-defined name (empty when not set).
};

// ============================================================================
// Enums matching AnalysisResults.Quantity and AnalysisResults.PeakType
// ============================================================================

/// Mirrors brimfile.AnalysisResults.Quantity.
enum class Quantity {
    Shift,
    Width,
    Amplitude,
    Offset,
    R2,
    RMSE,
    Cov_matrix,
    Elastic_contrast,
    Viscous_contrast,
};

/// Mirrors brimfile.AnalysisResults.PeakType.
enum class PeakType {
    AntiStokes,
    Stokes,
    Average,
};

// ============================================================================
// Image result (returned by BrimAnalysisResults::get_image)
// ============================================================================

/**
 * @brief Return value of BrimAnalysisResults::get_image().
 *
 * `image` contains the spatial map for the requested quantity.  `px_size` is
 * the pixel size along the z, y, x axes (index 0, 1, 2 respectively).
 */
struct ImageResult {
    NDArray image;
    std::array<MetadataItem, 3> px_size; ///< [z, y, x] pixel size.
};

// ============================================================================
// Spectrum result (returned by BrimData::get_spectrum_in_image and
//                              BrimData::get_PSD_as_spatial_map)
// ============================================================================

/**
 * @brief Return value of BrimData::get_spectrum_in_image() and
 *        BrimData::get_PSD_as_spatial_map().
 */
struct SpectrumResult {
    NDArray psd;            ///< Power Spectral Density array.
    NDArray frequency;      ///< Frequency array (same shape as psd or 1-D).
    std::string psd_units;  ///< Units of psd (may be empty).
    std::string freq_units; ///< Units of frequency (may be empty).
};

// ============================================================================
// Write data structures (used in BrimFile::create_data_group and
//                             BrimData::create_analysis_results_group)
// ============================================================================

/**
 * @brief Data for one peak type passed to create_analysis_results_group().
 *
 * Every NDArray field that is non-empty will be written to the file together
 * with its optional units string.
 */
struct AnalysisData {
    NDArray shift;
    std::string shift_units;
    NDArray width;
    std::string width_units;
    NDArray amplitude;
    std::string amplitude_units;
    NDArray offset;
    std::string offset_units;
    NDArray R2;
    NDArray RMSE;
};

// ============================================================================
// Forward declarations
// ============================================================================

class BrimFile;
class BrimData;
class BrimAnalysisResults;
class BrimMetadata;

// ============================================================================
// Internal detail – opaque handle owning a PyObject*
// ============================================================================

namespace detail {

/// RAII wrapper for a Python object reference.
class PyObj {
public:
    PyObj() noexcept : ptr_(nullptr) {}
    explicit PyObj(PyObject* p) noexcept : ptr_(p) {}
    ~PyObj() { Py_XDECREF(ptr_); }

    // Non-copyable, movable.
    PyObj(const PyObj&) = delete;
    PyObj& operator=(const PyObj&) = delete;
    PyObj(PyObj&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
    PyObj& operator=(PyObj&& o) noexcept {
        if (this != &o) { Py_XDECREF(ptr_); ptr_ = o.ptr_; o.ptr_ = nullptr; }
        return *this;
    }

    PyObject* get() const noexcept { return ptr_; }

    /// Release ownership without decrementing the reference count.
    PyObject* release() noexcept {
        PyObject* p = ptr_;
        ptr_ = nullptr;
        return p;
    }

    /// Borrow a new reference (increments the reference count).
    static PyObj borrow(PyObject* p) {
        Py_XINCREF(p);
        return PyObj(p);
    }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    PyObject* ptr_;
};

} // namespace detail

// ============================================================================
// PythonRuntime – manages the CPython interpreter lifecycle
// ============================================================================

/**
 * @brief Manages the lifecycle of the embedded CPython interpreter.
 *
 * Call PythonRuntime::init() once before creating any brimfile objects.
 * The interpreter is finalized automatically when the process exits.
 *
 * If the caller has already initialized Python (e.g. when embedding Python
 * from MATLAB or another host), init() detects this and skips
 * Py_Initialize(), leaving ownership with the host.
 */
class PythonRuntime {
public:
    /**
     * @brief Initialize (or attach to) the CPython interpreter.
     *
     * Safe to call multiple times; subsequent calls are no-ops.
     *
     * @throws PythonError if the brimfile or numpy Python packages cannot be
     *         imported.
     */
    static void init();

    /// Returns true when the interpreter was successfully initialized.
    static bool is_initialized() noexcept { return initialized_; }

private:
    static bool initialized_;
    static bool we_own_interpreter_;
};

// ============================================================================
// BrimMetadata
// ============================================================================

/**
 * @brief C++ wrapper for brimfile.Metadata.
 *
 * Access experiment, optics, sample, and Brillouin metadata stored in a brim
 * file.
 */
class BrimMetadata {
public:
    /// Metadata type groups mirroring brimfile.Metadata.Type.
    enum class Type {
        Experiment,
        Optics,
        Brillouin,
        Acquisition,
        Spectrometer,
    };

    /**
     * @brief Retrieve a single metadata attribute.
     *
     * @param key Key in the format "Group.Field", e.g. "Experiment.Datetime".
     * @return MetadataItem containing the value and units.
     * @throws PythonError if the key is not found or the format is invalid.
     */
    MetadataItem operator[](const std::string& key) const;

    /**
     * @brief Return all metadata attributes for a given type as a map.
     *
     * @param type Metadata group type.
     * @return Map from attribute name to MetadataItem.
     * @throws PythonError on Python errors.
     */
    std::map<std::string, MetadataItem> to_dict(Type type) const;

    /**
     * @brief Return all metadata attributes for every type.
     *
     * @return Map from type name to a map of attribute name → MetadataItem.
     * @throws PythonError on Python errors.
     */
    std::map<std::string, std::map<std::string, MetadataItem>> all_to_dict() const;

    /**
     * @brief Add metadata attributes.
     *
     * @param type  Metadata group type.
     * @param items Map from attribute name to MetadataItem.
     * @param local If true, attributes are stored in the specific data group;
     *              otherwise they are stored in the file-level metadata group.
     * @throws PythonError on Python errors.
     */
    void add(Type type,
             const std::map<std::string, MetadataItem>& items,
             bool local = false);

    // Internal constructor used by BrimData.
    explicit BrimMetadata(detail::PyObj py_obj) : py_obj_(std::move(py_obj)) {}

private:
    detail::PyObj py_obj_;
};

// ============================================================================
// BrimAnalysisResults
// ============================================================================

/**
 * @brief C++ wrapper for brimfile.AnalysisResults.
 *
 * Provides access to the spatial maps of quantities derived from spectral
 * fitting (shift, width, amplitude, etc.) stored in a brim file.
 */
class BrimAnalysisResults {
public:
    /// Returns the internal name of this analysis results group.
    std::string get_name() const;

    /**
     * @brief Retrieve the units string for a given quantity and peak type.
     *
     * @param qt    Quantity of interest.
     * @param pt    Peak type (default: AntiStokes).
     * @param index Peak index for multi-peak fits (default: 0).
     * @return Optional units string; empty optional when no units are defined.
     * @throws PythonError on Python errors.
     */
    std::optional<std::string> get_units(Quantity qt,
                                         PeakType pt = PeakType::AntiStokes,
                                         int index = 0) const;

    /**
     * @brief Retrieve a spatial image for a given quantity and peak type.
     *
     * @param qt    Quantity of interest.
     * @param pt    Peak type (default: AntiStokes).
     * @param index Peak index for multi-peak fits (default: 0).
     * @return ImageResult containing the image array and the pixel size.
     * @throws PythonError on Python errors.
     */
    ImageResult get_image(Quantity qt,
                          PeakType pt = PeakType::AntiStokes,
                          int index = 0) const;

    /**
     * @brief Retrieve the value of a quantity at a specific pixel.
     *
     * @param coord z, y, x spatial coordinates as a 3-element array.
     * @param qt    Quantity of interest.
     * @param pt    Peak type (default: AntiStokes).
     * @param index Peak index for multi-peak fits (default: 0).
     * @return NDArray containing the scalar or multi-dimensional result.
     * @throws PythonError on Python errors.
     */
    NDArray get_quantity_at_pixel(const std::array<int, 3>& coord,
                                  Quantity qt,
                                  PeakType pt = PeakType::AntiStokes,
                                  int index = 0) const;

    /**
     * @brief List the peak types present for a given peak index.
     *
     * @param index Peak index (default: 0).
     * @return Vector of peak type names (e.g. {"AS", "S"}).
     * @throws PythonError on Python errors.
     */
    std::vector<std::string> list_existing_peak_types(int index = 0) const;

    /**
     * @brief List the quantities present for a given peak type and index.
     *
     * @param pt    Peak type (default: AntiStokes).
     * @param index Peak index (default: 0).
     * @return Vector of quantity names (e.g. {"Shift", "Width"}).
     * @throws PythonError on Python errors.
     */
    std::vector<std::string> list_existing_quantities(
        PeakType pt = PeakType::AntiStokes,
        int index = 0) const;

    /**
     * @brief Returns the fit model name used for this analysis group.
     *
     * @return Fit model string (e.g. "Lorentzian"), or empty string if unknown.
     */
    std::string fit_model() const;

    // Internal constructor used by BrimData.
    explicit BrimAnalysisResults(detail::PyObj py_obj)
        : py_obj_(std::move(py_obj)) {}

private:
    detail::PyObj py_obj_;
};

// ============================================================================
// BrimData
// ============================================================================

/**
 * @brief C++ wrapper for brimfile.Data.
 *
 * Represents a single data group within a brim file.  Use BrimFile::get_data()
 * to obtain an instance.
 */
class BrimData {
public:
    /// Returns the internal name of this data group.
    std::string get_name() const;

    /// Returns the numeric index of this data group.
    int get_index() const;

    /**
     * @brief Retrieve the full PSD spatial map and the corresponding frequency.
     *
     * For non-sparse data the returned PSD has shape (z, y, x, spectrum).
     * For sparse data the spectra are reindexed into a Cartesian grid using the
     * spatial map stored in the file.
     *
     * @param broadcast_frequency When true (default), the frequency array is
     *        broadcast to match the shape of the PSD even when the file stores a
     *        single shared frequency axis.
     * @return SpectrumResult containing psd, frequency, and their units.
     * @throws PythonError on Python errors.
     */
    SpectrumResult get_PSD_as_spatial_map(bool broadcast_frequency = true) const;

    /**
     * @brief Retrieve the spectrum at a specific spatial coordinate.
     *
     * @param coord z, y, x spatial coordinates as a 3-element array.
     * @return SpectrumResult for the specified pixel.
     * @throws PythonError on Python errors.
     */
    SpectrumResult get_spectrum_in_image(const std::array<int, 3>& coord) const;

    /**
     * @brief Return the metadata object associated with this data group.
     *
     * @return BrimMetadata instance.
     * @throws PythonError on Python errors.
     */
    BrimMetadata get_metadata() const;

    /**
     * @brief List all analysis results groups in this data group.
     *
     * @param retrieve_custom_name When true the custom user-defined name is
     *        populated in GroupInfo::custom_name.
     * @return Vector of GroupInfo descriptors ordered by index.
     * @throws PythonError on Python errors.
     */
    std::vector<GroupInfo> list_analysis_results(
        bool retrieve_custom_name = false) const;

    /**
     * @brief Return the analysis results group at a given index.
     *
     * @param index Numeric index (default: 0).
     * @return BrimAnalysisResults instance.
     * @throws PythonError if the index does not exist.
     */
    BrimAnalysisResults get_analysis_results(int index = 0) const;

    /**
     * @brief Create a new analysis results group and write data into it.
     *
     * @param data_as Analysis data for the AntiStokes peak (required).
     * @param data_s  Analysis data for the Stokes peak (optional; pass a
     *                default-constructed AnalysisData to omit).
     * @param name      Optional custom name.
     * @param fit_model Optional fit-model name string (e.g. "Lorentzian").
     * @return Newly created BrimAnalysisResults instance.
     * @throws PythonError on Python errors.
     */
    BrimAnalysisResults create_analysis_results_group(
        const AnalysisData& data_as,
        const AnalysisData& data_s = AnalysisData{},
        const std::string& name = "",
        const std::string& fit_model = "") const;

    // Internal constructor used by BrimFile.
    explicit BrimData(detail::PyObj py_obj) : py_obj_(std::move(py_obj)) {}

private:
    detail::PyObj py_obj_;
};

// ============================================================================
// BrimFile
// ============================================================================

/**
 * @brief C++ wrapper for brimfile.File.
 *
 * Entry point for reading and writing brim files.
 *
 * @code
 *   brimfile::PythonRuntime::init();
 *   brimfile::BrimFile f("data.brim.zarr");
 *   auto d = f.get_data(0);
 *   f.close();
 * @endcode
 */
class BrimFile {
public:
    /**
     * @brief Open an existing brim file.
     *
     * @param filename Path to the brim file.
     * @param mode     File open mode ('r', 'r+', 'a', 'w', 'w-').
     *                 See brimfile.File for details.  Default is 'r'
     *                 (read-only).
     * @throws PythonError if the file cannot be opened or is invalid.
     */
    explicit BrimFile(const std::string& filename,
                      const std::string& mode = "r");

    /**
     * @brief Create a new brim file.
     *
     * Fails if the file already exists.
     *
     * @param filename Path for the new brim file.
     * @return BrimFile instance with read/write access.
     * @throws PythonError if the file cannot be created.
     */
    static BrimFile create(const std::string& filename);

    /// Destructor – calls close() if the file is still open.
    ~BrimFile();

    // Non-copyable.
    BrimFile(const BrimFile&) = delete;
    BrimFile& operator=(const BrimFile&) = delete;

    // Movable.
    BrimFile(BrimFile&&) noexcept;
    BrimFile& operator=(BrimFile&&) noexcept;

    /**
     * @brief Close the file and release all associated Python resources.
     *
     * This method is idempotent; calling it multiple times is safe.
     */
    void close();

    /**
     * @brief Returns true if the file was opened in read-only mode.
     * @throws PythonError on Python errors.
     */
    bool is_read_only() const;

    /**
     * @brief Returns the file path.
     * @throws PythonError on Python errors.
     */
    std::string filename() const;

    /**
     * @brief List all data groups stored in the file.
     *
     * @param retrieve_custom_name When true the custom user-defined name is
     *        populated in GroupInfo::custom_name.
     * @return Vector of GroupInfo descriptors ordered by index.
     * @throws PythonError on Python errors.
     */
    std::vector<GroupInfo> list_data_groups(
        bool retrieve_custom_name = false) const;

    /**
     * @brief Return the data group at a given index.
     *
     * @param index Numeric index (default: 0).
     * @return BrimData instance.
     * @throws PythonError if the index does not exist.
     */
    BrimData get_data(int index = 0) const;

    /**
     * @brief Add a new dense data group to the file.
     *
     * @param psd        PSD data with shape (z, y, x, spectrum).
     * @param frequency  Frequency data broadcastable to psd shape.
     * @param px_size_um Pixel size in µm for the z, y, x axes (3 elements;
     *                   use NaN or 0 for unused axes).
     * @param name       Optional custom name for the data group.
     * @return Newly created BrimData instance.
     * @throws PythonError on Python errors or invalid data.
     */
    BrimData create_data_group(const NDArray& psd,
                               const NDArray& frequency,
                               const std::array<double, 3>& px_size_um,
                               const std::string& name = "") const;

private:
    detail::PyObj py_obj_;

    // Private constructor used by create().
    explicit BrimFile(detail::PyObj py_obj) : py_obj_(std::move(py_obj)) {}
};

} // namespace brimfile
