"""
Unit tests for the AnalysisResults class in brimfile.
"""


import numpy as np

import brimfile as brim


class TestAnalysisResultsProperties:
    """Tests for AnalysisResults properties."""
    
    def test_get_name(self, simple_brim_file):
        """Test getting the name of analysis results."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        name = ar.get_name()
        assert name is not None
        assert isinstance(name, str)
        f.close()
    
    def test_fit_model_property(self, simple_brim_file):
        """Test accessing fit model property."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        fit_model = ar.fit_model
        assert fit_model is not None
        assert fit_model == brim.Data.AnalysisResults.FitModel.Lorentzian
        f.close()


class TestPeakTypesAndQuantities:
    """Tests for peak types and quantities in AnalysisResults."""
    
    def test_list_existing_peak_types(self, simple_brim_file):
        """Test listing existing peak types."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        peak_types = ar.list_existing_peak_types()
        assert peak_types is not None
        assert isinstance(peak_types, (list, tuple))
        assert len(peak_types) > 0
        f.close()
    
    def test_list_existing_quantities(self, simple_brim_file):
        """Test listing existing quantities."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        quantities = ar.list_existing_quantities()
        assert quantities is not None
        assert isinstance(quantities, (list, tuple))
        assert len(quantities) > 0
        f.close()
    
    def test_peak_type_enum_values(self):
        """Test PeakType enum values."""
        PeakType = brim.Data.AnalysisResults.PeakType
        assert PeakType.average is not None
        assert PeakType.Stokes is not None
        assert PeakType.AntiStokes is not None
    
    def test_quantity_enum_values(self):
        """Test Quantity enum values."""
        Quantity = brim.Data.AnalysisResults.Quantity
        assert Quantity.Shift is not None
        assert Quantity.Width is not None


class TestImageRetrieval:
    """Tests for retrieving images from AnalysisResults."""
    
    def test_get_image_shift(self, simple_brim_file):
        """Test getting shift image."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        img, px_size = ar.get_image(Quantity.Shift, PeakType.average)
        
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert px_size is not None
        assert len(px_size) == 3
        f.close()
    
    def test_get_image_width(self, simple_brim_file):
        """Test getting width image."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        img, px_size = ar.get_image(Quantity.Width, PeakType.average)
        
        assert img is not None
        assert isinstance(img, np.ndarray)
        f.close()
    
    def test_image_dimensions(self, simple_brim_file, sample_data):
        """Test that image dimensions match expected dimensions."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        img, _ = ar.get_image(Quantity.Shift, PeakType.average)
        
        # Should match the dimensions from sample_data
        Nz, Ny, Nx = sample_data['dimensions']
        assert img.shape == (Nz, Ny, Nx)
        f.close()


class TestUnitsRetrieval:
    """Tests for retrieving units from AnalysisResults."""
    
    def test_get_units_shift(self, simple_brim_file):
        """Test getting units for shift quantity."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        units = ar.get_units(Quantity.Shift)
        
        assert units is not None
        assert isinstance(units, str)
        assert units == 'GHz'
        f.close()
    
    def test_get_units_width(self, simple_brim_file):
        """Test getting units for width quantity."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        units = ar.get_units(Quantity.Width)
        
        assert units is not None
        assert isinstance(units, str)
        f.close()


class TestPixelQuantityRetrieval:
    """Tests for retrieving quantities at specific pixels."""
    
    def test_get_quantity_at_pixel(self, simple_brim_file):
        """Test getting quantity value at a specific pixel."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        coord = (1, 3, 4)
        value = ar.get_quantity_at_pixel(coord, Quantity.Shift, PeakType.average)
        
        assert value is not None
        assert isinstance(value, (int, float, np.number))
        f.close()
    
    def test_quantity_at_pixel_matches_image(self, simple_brim_file):
        """Test that quantity at pixel matches image value."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        coord = (1, 3, 4)
        img, _ = ar.get_image(Quantity.Shift, PeakType.average)
        qt_at_px = ar.get_quantity_at_pixel(coord, Quantity.Shift, PeakType.average)
        
        assert img[coord] == qt_at_px
        f.close()
    
    def test_get_quantity_at_different_pixels(self, simple_brim_file):
        """Test getting quantities at multiple different pixels."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        Quantity = brim.Data.AnalysisResults.Quantity
        PeakType = brim.Data.AnalysisResults.PeakType
        
        coords = [(0, 0, 0), (1, 2, 3), (2, 3, 5)]
        for coord in coords:
            value = ar.get_quantity_at_pixel(coord, Quantity.Shift, PeakType.average)
            assert value is not None
        
        f.close()


class TestAnalysisResultsCreation:
    """Tests for creating analysis results."""
    
    def test_create_with_custom_name(self, empty_brim_file, sample_data):
        """Test creating analysis results with custom name."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        
        ar = data.create_analysis_results_group(
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            },
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            },
            name='custom_analysis'
        )
        
        name = ar.get_name()
        assert 'custom_analysis' in name or name == 'custom_analysis'
        f.close()
    
    def test_create_with_fit_model(self, empty_brim_file, sample_data):
        """Test creating analysis results with fit model."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        
        ar = data.create_analysis_results_group(
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            },
            {
                'shift': sample_data['shift'],
                'shift_units': 'GHz',
                'width': sample_data['width'],
                'width_units': 'GHz'
            },
            fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian
        )
        
        assert ar.fit_model == brim.Data.AnalysisResults.FitModel.Lorentzian
        f.close()


class TestFitModel:
    """Tests for FitModel enum."""
    
    def test_fit_model_enum_values(self):
        """Test that FitModel enum has expected values."""
        FitModel = brim.Data.AnalysisResults.FitModel
        assert FitModel.Lorentzian is not None
        assert FitModel.Voigt is not None
        assert FitModel.DHO is not None
