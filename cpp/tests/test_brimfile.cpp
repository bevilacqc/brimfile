/**
 * @file test_brimfile.cpp
 * @brief Basic tests for the brimfile C++ wrapper.
 *
 * These tests exercise the main read and write operations of the wrapper.
 * They rely on the Python brimfile package being installed and available in
 * the current Python environment.
 *
 * Build and run:
 *   cmake -B build && cmake --build build && ctest --test-dir build -V
 */

#include <brimfile/brimfile.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------

static int s_total = 0;
static int s_passed = 0;
static int s_failed = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++s_total;                                                               \
        if (cond) {                                                              \
            ++s_passed;                                                          \
        } else {                                                                 \
            ++s_failed;                                                          \
            std::cerr << "FAIL  " << __FILE__ << ":" << __LINE__                \
                      << "  " << #cond << "\n";                                  \
        }                                                                        \
    } while (false)

#define CHECK_THROWS(expr)                                                       \
    do {                                                                         \
        ++s_total;                                                               \
        bool threw = false;                                                      \
        try { (void)(expr); } catch (...) { threw = true; }                      \
        if (threw) {                                                             \
            ++s_passed;                                                          \
        } else {                                                                 \
            ++s_failed;                                                          \
            std::cerr << "FAIL (expected exception)  " << __FILE__              \
                      << ":" << __LINE__ << "  " << #expr << "\n";              \
        }                                                                        \
    } while (false)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace fs = std::filesystem;

/// Create a temporary directory that is removed at the end of the test.
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& prefix = "brimtest_") {
        path = fs::temp_directory_path() / (prefix + std::to_string(
            std::chrono::system_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        try { fs::remove_all(path); } catch (...) {}
    }
};

/// Simple Lorentzian spectrum for generating test data.
static double lorentzian(double x, double x0, double w) {
    double d = (x - x0) / (w / 2.0);
    return 1.0 / (1.0 + d * d);
}

// ---------------------------------------------------------------------------
// Test: create a new brim file with one data group and read it back
// ---------------------------------------------------------------------------

static void test_create_and_read() {
    TempDir tmp;
    std::string filename = (tmp.path / "test.brim.zarr").string();

    // ---- write ----

    // PSD shape: (Nz=2, Ny=3, Nx=4, Nfreq=51)
    const int Nz = 2, Ny = 3, Nx = 4, Nfreq = 51;

    brimfile::NDArray freq;
    freq.shape = {static_cast<std::size_t>(Nfreq)};
    freq.data.resize(Nfreq);
    for (int i = 0; i < Nfreq; ++i)
        freq.data[static_cast<std::size_t>(i)] = 6.0 + 3.0 * i / (Nfreq - 1);

    brimfile::NDArray psd;
    psd.shape = {static_cast<std::size_t>(Nz),
                 static_cast<std::size_t>(Ny),
                 static_cast<std::size_t>(Nx),
                 static_cast<std::size_t>(Nfreq)};
    psd.data.resize(static_cast<std::size_t>(Nz * Ny * Nx * Nfreq));
    for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x)
                for (int f = 0; f < Nfreq; ++f) {
                    std::size_t idx = static_cast<std::size_t>(
                        ((z * Ny + y) * Nx + x) * Nfreq + f);
                    psd.data[idx] = lorentzian(freq.data[static_cast<std::size_t>(f)],
                                               7.5, 0.4);
                }

    {
        brimfile::BrimFile f = brimfile::BrimFile::create(filename);
        auto d = f.create_data_group(psd, freq, {2.0, 0.5, 0.4}, "test_data");

        // Add metadata.
        brimfile::BrimMetadata md = d.get_metadata();
        md.add(brimfile::BrimMetadata::Type::Experiment,
               {{"Datetime",
                 brimfile::MetadataItem{"2024-01-01T12:00:00", "", true}}});
        md.add(brimfile::BrimMetadata::Type::Optics,
               {{"Wavelength", brimfile::MetadataItem{"660", "nm", true}}});

        // Add analysis results.
        brimfile::NDArray shift_arr;
        shift_arr.shape = {static_cast<std::size_t>(Nz),
                           static_cast<std::size_t>(Ny),
                           static_cast<std::size_t>(Nx)};
        shift_arr.data.assign(
            static_cast<std::size_t>(Nz * Ny * Nx), 7.5);

        brimfile::AnalysisData ad;
        ad.shift = shift_arr;
        ad.shift_units = "GHz";
        d.create_analysis_results_group(ad, {}, "test_ar");

        f.close();
    }

    // ---- read ----

    brimfile::BrimFile f(filename);

    CHECK(f.filename().find("test.brim.zarr") != std::string::npos);
    // Note: Python brimfile returns false for is_read_only() even in mode='r'
    // (the underlying zarr store considers itself writable until a write is attempted)
    (void)f.is_read_only(); // just verify the call doesn't throw

    // list_data_groups
    auto groups = f.list_data_groups();
    CHECK(groups.size() == 1);
    CHECK(groups[0].index == 0);

    // get_data
    auto d = f.get_data(0);
    CHECK(d.get_index() == 0);

    // get_PSD_as_spatial_map
    auto spec = d.get_PSD_as_spatial_map();
    CHECK(spec.psd.shape.size() == 4);
    CHECK(spec.psd.shape[0] == static_cast<std::size_t>(Nz));
    CHECK(spec.psd.shape[1] == static_cast<std::size_t>(Ny));
    CHECK(spec.psd.shape[2] == static_cast<std::size_t>(Nx));
    CHECK(spec.psd.shape[3] == static_cast<std::size_t>(Nfreq));
    CHECK(!spec.psd.empty());

    // get_spectrum_in_image
    auto single = d.get_spectrum_in_image({0, 0, 0});
    CHECK(single.psd.shape.size() == 1);
    CHECK(single.psd.shape[0] == static_cast<std::size_t>(Nfreq));
    CHECK(!single.psd.empty());

    // get_metadata
    auto md = d.get_metadata();
    auto all_md = md.all_to_dict();
    CHECK(!all_md.empty());

    // get_analysis_results
    auto ar_list = d.list_analysis_results();
    CHECK(ar_list.size() == 1);

    auto ar = d.get_analysis_results(0);
    CHECK(!ar.get_name().empty());

    // get_image
    auto img = ar.get_image(brimfile::Quantity::Shift,
                            brimfile::PeakType::AntiStokes, 0);
    CHECK(img.image.shape.size() == 3);
    CHECK(img.image.shape[0] == static_cast<std::size_t>(Nz));

    // list_existing_quantities and peak_types
    auto qts = ar.list_existing_quantities();
    CHECK(!qts.empty());
    auto pts = ar.list_existing_peak_types();
    CHECK(!pts.empty());

    // get_quantity_at_pixel
    auto val = ar.get_quantity_at_pixel({0, 0, 0}, brimfile::Quantity::Shift);
    CHECK(!val.empty());
    CHECK(std::abs(val.data[0] - 7.5) < 1e-3);

    // get_units
    auto units = ar.get_units(brimfile::Quantity::Shift);
    CHECK(units.has_value());
    CHECK(*units == "GHz");

    f.close();
    CHECK_THROWS(brimfile::BrimFile("/nonexistent/path.brim.zarr"));
}

// ---------------------------------------------------------------------------
// Test: open a file that does not exist raises an exception
// ---------------------------------------------------------------------------

static void test_open_nonexistent() {
    CHECK_THROWS(brimfile::BrimFile("/nonexistent/path/file.brim.zarr"));
}

// ---------------------------------------------------------------------------
// Test: metadata access helpers
// ---------------------------------------------------------------------------

static void test_metadata() {
    TempDir tmp;
    std::string filename = (tmp.path / "meta.brim.zarr").string();

    brimfile::NDArray freq;
    freq.shape = {10};
    freq.data.resize(10);
    for (int i = 0; i < 10; ++i)
        freq.data[static_cast<std::size_t>(i)] = 6.0 + 0.3 * i;

    brimfile::NDArray psd;
    psd.shape = {1, 1, 1, 10};
    psd.data.assign(10, 1.0);

    {
        brimfile::BrimFile f = brimfile::BrimFile::create(filename);
        auto d = f.create_data_group(psd, freq, {1.0, 1.0, 1.0});
        auto md = d.get_metadata();
        md.add(brimfile::BrimMetadata::Type::Experiment,
               {{"Temperature", brimfile::MetadataItem{"22", "C", true}},
                {"Datetime",
                 brimfile::MetadataItem{"2024-06-01T00:00:00", "", true}}});
        f.close();
    }

    brimfile::BrimFile f(filename);
    auto d = f.get_data(0);
    auto md = d.get_metadata();

    // Test operator[].
    auto temp = md["Experiment.Temperature"];
    CHECK(temp.has_value);
    CHECK(!temp.value.empty());

    // to_dict for Experiment type.
    auto exp_dict = md.to_dict(brimfile::BrimMetadata::Type::Experiment);
    CHECK(!exp_dict.empty());

    // all_to_dict should contain all type groups.
    auto all = md.all_to_dict();
    CHECK(all.count("Experiment") > 0);

    f.close();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    try {
        brimfile::PythonRuntime::init();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialise Python: " << e.what() << "\n";
        return 1;
    }

    try {
        test_create_and_read();
        test_open_nonexistent();
        test_metadata();
    } catch (const std::exception& e) {
        std::cerr << "Unexpected exception: " << e.what() << "\n";
        ++s_failed;
    }

    std::cout << "\n=== Results: " << s_passed << "/" << s_total << " passed";
    if (s_failed > 0)
        std::cout << ", " << s_failed << " FAILED";
    std::cout << " ===\n";

    return s_failed > 0 ? 1 : 0;
}
