/**
 * @file example.cpp
 * @brief Example showing how to read and write brim files from C++.
 *
 * This example demonstrates the main read/write operations supported by the
 * brimfile C++ wrapper.
 *
 * Build (from the cpp/ directory):
 *   cmake -B build && cmake --build build
 *   ./build/examples/brimfile_example
 */

#include <brimfile/brimfile.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: generate a synthetic Lorentzian spectrum
// ---------------------------------------------------------------------------

static double lorentzian(double x, double x0, double w) {
    double d = (x - x0) / (w / 2.0);
    return 1.0 / (1.0 + d * d);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------
    // Initialise the embedded Python interpreter.
    // This must be called once before using any brimfile:: functions.
    // ------------------------------------------------------------------
    try {
        brimfile::PythonRuntime::init();
    } catch (const brimfile::PythonError& e) {
        std::cerr << "Error initialising Python: " << e.what() << "\n";
        std::cerr << "Make sure the brimfile Python package is installed.\n";
        return 1;
    }

    const std::string filename = "example_output.brim.zarr";

    // ==================================================================
    // Writing: create a new brim file
    // ==================================================================
    {
        std::cout << "Creating " << filename << " ...\n";

        // ---- synthetic data ----
        const int Nz = 3, Ny = 5, Nx = 7, Nfreq = 151;
        const double shift_ghz = 7.5, width_ghz = 0.4;

        brimfile::NDArray freq;
        freq.shape = {static_cast<std::size_t>(Nfreq)};
        freq.data.resize(static_cast<std::size_t>(Nfreq));
        for (int i = 0; i < Nfreq; ++i)
            freq.data[static_cast<std::size_t>(i)] =
                6.0 + 3.0 * i / (Nfreq - 1);

        brimfile::NDArray psd;
        psd.shape = {static_cast<std::size_t>(Nz),
                     static_cast<std::size_t>(Ny),
                     static_cast<std::size_t>(Nx),
                     static_cast<std::size_t>(Nfreq)};
        psd.data.resize(
            static_cast<std::size_t>(Nz * Ny * Nx * Nfreq));
        for (int z = 0; z < Nz; ++z)
            for (int y = 0; y < Ny; ++y)
                for (int x = 0; x < Nx; ++x)
                    for (int f = 0; f < Nfreq; ++f) {
                        std::size_t idx = static_cast<std::size_t>(
                            ((z * Ny + y) * Nx + x) * Nfreq + f);
                        psd.data[idx] =
                            lorentzian(freq.data[static_cast<std::size_t>(f)],
                                       shift_ghz, width_ghz);
                    }

        // ---- create file ----
        try {
            brimfile::BrimFile f = brimfile::BrimFile::create(filename);

            // Add data group (pixel size in µm: dz=2, dy=0.5, dx=0.4).
            auto d = f.create_data_group(psd, freq, {2.0, 0.5, 0.4},
                                         "example_data");

            // Add metadata.
            auto md = d.get_metadata();
            md.add(brimfile::BrimMetadata::Type::Experiment,
                   {{"Datetime",
                     brimfile::MetadataItem{
                         "2024-01-01T12:00:00", "", true}},
                    {"Temperature",
                     brimfile::MetadataItem{"22", "C", true}}});
            md.add(brimfile::BrimMetadata::Type::Optics,
                   {{"Wavelength",
                     brimfile::MetadataItem{"660", "nm", true}}});
            md.add(brimfile::BrimMetadata::Type::Brillouin,
                   {{"Scattering_angle",
                     brimfile::MetadataItem{"180", "deg", true}}});

            // Add analysis results (uniform shift/width maps).
            brimfile::NDArray shift_map;
            shift_map.shape = {static_cast<std::size_t>(Nz),
                               static_cast<std::size_t>(Ny),
                               static_cast<std::size_t>(Nx)};
            shift_map.data.assign(
                static_cast<std::size_t>(Nz * Ny * Nx), shift_ghz);

            brimfile::NDArray width_map;
            width_map.shape = shift_map.shape;
            width_map.data.assign(
                static_cast<std::size_t>(Nz * Ny * Nx), width_ghz);

            brimfile::AnalysisData ad;
            ad.shift       = shift_map;
            ad.shift_units = "GHz";
            ad.width       = width_map;
            ad.width_units = "GHz";

            d.create_analysis_results_group(
                ad, ad, "example_analysis", "Lorentzian");

            f.close();
            std::cout << "File written successfully.\n";
        } catch (const brimfile::Error& e) {
            std::cerr << "Write error: " << e.what() << "\n";
            return 1;
        }
    }

    // ==================================================================
    // Reading: open the file we just created
    // ==================================================================
    {
        std::cout << "\nOpening " << filename << " ...\n";
        try {
            brimfile::BrimFile f(filename);

            std::cout << "  filename    : " << f.filename() << "\n";
            std::cout << "  read-only   : "
                      << (f.is_read_only() ? "yes" : "no") << "\n";

            // List data groups.
            auto groups = f.list_data_groups(true);
            std::cout << "  data groups : " << groups.size() << "\n";
            for (const auto& g : groups)
                std::cout << "    [" << g.index << "] " << g.name << "\n";

            // Get the first data group.
            auto d = f.get_data(0);
            std::cout << "  data name   : " << d.get_name() << "\n";

            // Read the full PSD spatial map.
            auto spec = d.get_PSD_as_spatial_map();
            std::cout << "  PSD shape   : ";
            for (std::size_t i = 0; i < spec.psd.shape.size(); ++i) {
                if (i) std::cout << " x ";
                std::cout << spec.psd.shape[i];
            }
            std::cout << "  (" << spec.psd_units << ")\n";

            // Read a single spectrum.
            auto single = d.get_spectrum_in_image({0, 0, 0});
            std::cout << "  spectrum[0,0,0] length: "
                      << single.psd.shape[0] << "\n";

            // Read metadata.
            auto md = d.get_metadata();
            auto temp_item = md["Experiment.Temperature"];
            std::cout << "  temperature : " << temp_item.value
                      << " " << temp_item.units << "\n";

            // List analysis results.
            auto ar_list = d.list_analysis_results(true);
            std::cout << "  analysis results: " << ar_list.size() << "\n";

            // Read the first analysis results.
            auto ar = d.get_analysis_results(0);
            std::cout << "  AR name     : " << ar.get_name() << "\n";
            std::cout << "  fit model   : " << ar.fit_model() << "\n";

            // Get existing peak types and quantities.
            auto pts = ar.list_existing_peak_types();
            std::cout << "  peak types  : ";
            for (const auto& pt : pts) std::cout << pt << " ";
            std::cout << "\n";

            auto qts = ar.list_existing_quantities();
            std::cout << "  quantities  : ";
            for (const auto& qt : qts) std::cout << qt << " ";
            std::cout << "\n";

            // Get the shift image.
            auto img = ar.get_image(brimfile::Quantity::Shift,
                                    brimfile::PeakType::AntiStokes);
            std::cout << "  shift image shape: ";
            for (std::size_t i = 0; i < img.image.shape.size(); ++i) {
                if (i) std::cout << " x ";
                std::cout << img.image.shape[i];
            }
            auto units = ar.get_units(brimfile::Quantity::Shift);
            if (units) std::cout << "  (" << *units << ")";
            std::cout << "\n";

            // Get shift value at pixel (0,0,0).
            auto val = ar.get_quantity_at_pixel(
                {0, 0, 0}, brimfile::Quantity::Shift);
            std::cout << "  shift[0,0,0]: " << val.data[0] << " GHz\n";

            f.close();
        } catch (const brimfile::Error& e) {
            std::cerr << "Read error: " << e.what() << "\n";
            return 1;
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
