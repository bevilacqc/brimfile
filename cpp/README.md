# brimfile C++ Wrapper

A C++ wrapper for the [brimfile](https://github.com/prevedel-lab/brimfile) Python package that lets you read and write Brillouin microscopy data files (`.brim.zarr`) directly from C++ applications.

## Approach

The wrapper uses **Python embedding** via the CPython C API
(`https://docs.python.org/3/extending/embedding.html`).  The brimfile
Python package is invoked through the embedded interpreter, which means:

- No reimplementation of the file format logic in C++.
- Full access to the complete brimfile Python API.
- Automatic NumPy array ↔ `NDArray` conversion using the [buffer
  protocol](https://docs.python.org/3/c-api/buffer.html).
- C++ exceptions (`brimfile::PythonError`) are raised whenever the
  Python layer raises an exception.

## Requirements

| Dependency | Version |
|---|---|
| Python development headers | 3.11+ |
| brimfile Python package | installed in the active environment |
| NumPy | installed in the active environment |
| CMake | 3.15+ |
| C++ compiler | C++17 or later |

Install brimfile from the repository root:
```bash
pip install -e .
```

## Building

```bash
cd cpp/
cmake -B build
cmake --build build
```

CMake automatically locates the Python interpreter and development libraries
via `find_package(Python3)`.  If you have multiple Python environments, set
`Python3_ROOT_DIR` or `Python3_EXECUTABLE` to point at the right one.

### CMake options

| Option | Default | Description |
|---|---|---|
| `BRIMFILE_CPP_BUILD_TESTS` | `ON` | Build the test binary |
| `BRIMFILE_CPP_BUILD_EXAMPLES` | `ON` | Build the example binary |

## Running tests

```bash
ctest --test-dir build -V
# or directly:
./build/tests/test_brimfile
```

## Quick start

```cpp
#include <brimfile/brimfile.hpp>
#include <iostream>

int main() {
    // Initialise the embedded Python interpreter once.
    brimfile::PythonRuntime::init();

    // ---- Reading ----
    brimfile::BrimFile f("data.brim.zarr");

    auto groups = f.list_data_groups();
    auto d      = f.get_data(0);

    // Full PSD spatial map: shape (z, y, x, spectrum)
    auto [psd, freq, psd_units, freq_units] = d.get_PSD_as_spatial_map();

    // Single spectrum
    auto spec = d.get_spectrum_in_image({0, 0, 0});

    // Metadata
    auto md   = d.get_metadata();
    auto temp = md["Experiment.Temperature"];
    std::cout << "Temperature: " << temp.value << " " << temp.units << "\n";

    // Analysis results
    auto ar  = d.get_analysis_results(0);
    auto img = ar.get_image(brimfile::Quantity::Shift, brimfile::PeakType::AntiStokes);
    auto val = ar.get_quantity_at_pixel({0, 0, 0}, brimfile::Quantity::Shift);

    f.close();

    // ---- Writing ----
    brimfile::NDArray psd_write;
    psd_write.shape = {3, 5, 7, 151};
    psd_write.data.assign(3 * 5 * 7 * 151, 1.0);

    brimfile::NDArray freq_write;
    freq_write.shape = {151};
    freq_write.data.resize(151);
    for (int i = 0; i < 151; ++i)
        freq_write.data[i] = 6.0 + 3.0 * i / 150.0;

    brimfile::BrimFile fw = brimfile::BrimFile::create("output.brim.zarr");
    auto dw = fw.create_data_group(psd_write, freq_write,
                                   {2.0, 0.5, 0.4},    // pixel size (µm)
                                   "my_data");

    auto mdw = dw.get_metadata();
    mdw.add(brimfile::BrimMetadata::Type::Experiment,
            {{"Datetime", brimfile::MetadataItem{"2024-01-01T12:00:00", "", true}}});

    brimfile::AnalysisData ad;
    ad.shift.shape = {3, 5, 7};
    ad.shift.data.assign(3 * 5 * 7, 7.5);
    ad.shift_units = "GHz";
    dw.create_analysis_results_group(ad, {}, "my_analysis");

    fw.close();
    return 0;
}
```

## API reference

### `brimfile::PythonRuntime`

| Member | Description |
|---|---|
| `static void init()` | Initialise (or attach to) the CPython interpreter. Call once before using the library. |
| `static bool is_initialized()` | True after a successful `init()`. |

### `brimfile::BrimFile`

| Member | Description |
|---|---|
| `BrimFile(filename, mode="r")` | Open an existing brim file. |
| `static BrimFile create(filename)` | Create a new brim file. |
| `void close()` | Close the file (also called by the destructor). |
| `bool is_read_only()` | Returns true if the file is read-only. |
| `std::string filename()` | Returns the file path. |
| `std::vector<GroupInfo> list_data_groups(retrieve_custom_name=false)` | List all data groups. |
| `BrimData get_data(index=0)` | Get a data group by index. |
| `BrimData create_data_group(psd, frequency, px_size_um, name="")` | Add a dense data group. |

### `brimfile::BrimData`

| Member | Description |
|---|---|
| `std::string get_name()` | Internal group name. |
| `int get_index()` | Numeric index. |
| `SpectrumResult get_PSD_as_spatial_map(broadcast_frequency=true)` | Full PSD spatial map. |
| `SpectrumResult get_spectrum_in_image(coord)` | Single spectrum at `{z, y, x}`. |
| `BrimMetadata get_metadata()` | Metadata associated with this data group. |
| `std::vector<GroupInfo> list_analysis_results(retrieve_custom_name=false)` | List analysis results groups. |
| `BrimAnalysisResults get_analysis_results(index=0)` | Get analysis results by index. |
| `BrimAnalysisResults create_analysis_results_group(data_as, data_s, name, fit_model)` | Write analysis results. |

### `brimfile::BrimAnalysisResults`

| Member | Description |
|---|---|
| `std::string get_name()` | Internal group name. |
| `std::optional<std::string> get_units(qt, pt, index)` | Units string for a quantity. |
| `ImageResult get_image(qt, pt, index)` | Spatial image for a quantity. |
| `NDArray get_quantity_at_pixel(coord, qt, pt, index)` | Value at a specific pixel. |
| `std::vector<std::string> list_existing_peak_types(index)` | Present peak types (e.g. `{"AS", "S"}`). |
| `std::vector<std::string> list_existing_quantities(pt, index)` | Present quantities (e.g. `{"Shift", "Width"}`). |
| `std::string fit_model()` | Fit model name (e.g. `"Lorentzian"`). |

### `brimfile::BrimMetadata`

| Member | Description |
|---|---|
| `MetadataItem operator[](key)` | Get a single attribute, e.g. `md["Experiment.Temperature"]`. |
| `std::map<std::string, MetadataItem> to_dict(type)` | All attributes for a metadata type. |
| `std::map<std::string, std::map<std::string, MetadataItem>> all_to_dict()` | All attributes for all types. |
| `void add(type, items, local=false)` | Add metadata attributes. |

### Data types

```cpp
struct NDArray {
    std::vector<double> data;           // elements in C (row-major) order
    std::vector<std::size_t> shape;     // size of each dimension
    std::size_t size() const;           // total number of elements
    bool empty() const;                 // true when data is empty
};

struct MetadataItem {
    std::string value;      // string representation of the value
    std::string units;      // units string (empty when not defined)
    bool has_value = false; // false when the field is absent
};

struct GroupInfo {
    int index = 0;
    std::string name;
    std::string custom_name;
};

struct SpectrumResult {
    NDArray psd;
    NDArray frequency;
    std::string psd_units;
    std::string freq_units;
};

struct ImageResult {
    NDArray image;
    std::array<MetadataItem, 3> px_size; // z, y, x pixel size
};

struct AnalysisData {
    NDArray shift;        std::string shift_units;
    NDArray width;        std::string width_units;
    NDArray amplitude;    std::string amplitude_units;
    NDArray offset;       std::string offset_units;
    NDArray R2;
    NDArray RMSE;
};
```

### Enums

```cpp
enum class Quantity {
    Shift, Width, Amplitude, Offset, R2, RMSE, Cov_matrix,
    Elastic_contrast, Viscous_contrast
};

enum class PeakType { AntiStokes, Stokes, Average };
```

## Thread safety

The Python GIL must be held when calling any function in this library.  If
you use Python from multiple threads, acquire the GIL with
`PyGILState_Ensure` / `PyGILState_Release` around each call into the
library.

## Linking in your own CMake project

```cmake
add_subdirectory(path/to/brimfile/cpp)
target_link_libraries(my_app PRIVATE brimfile_cpp)
```
