import numpy as np
import asyncio

import warnings
from enum import Enum

from .file_abstraction import FileAbstraction, sync, _async_getitem, _gather_sync
from .utils import concatenate_paths, list_objects_matching_pattern, get_object_name, set_object_name
from .utils import var_to_singleton, np_array_to_smallest_int_type, _guess_chunks

from .metadata import Metadata

from numbers import Number

from . import units
from .constants import brim_obj_names

__docformat__ = "google"


class Data:
    """
    Represents a data group within the brim file.
    """

    def __init__(self, file: FileAbstraction, path: str):
        """
        Initialize the Data object. This constructor should not be called directly.

        Args:
            file (File): The parent File object.
            path (str): The path to the data group within the file.
        """
        self._file = file
        self._path = path
        self._group = sync(file.open_group(path))
        
        # Check if data is sparse
        self._is_sparse = self._check_if_sparse()

        self._spatial_map, self._spatial_map_px_size = self._load_spatial_mapping()

    def get_name(self):
        """
        Returns the name of the data group.
        """
        return sync(get_object_name(self._file, self._path))
    
    def get_index(self):
        """
        Returns the index of the data group.
        """
        return int(self._path.split('/')[-1].split('_')[-1])
    
    def _check_if_sparse(self) -> bool:
        """
        Check if the data is sparse (i.e., not on a regular 3D grid).
        
        Returns:
            bool: True if data is sparse, False otherwise. 
                  For backward compatibility with old files that don't have the Sparse attribute:
                  - If Sparse attribute exists, use its value
                  - If Sparse attribute doesn't exist, check if Cartesian_visualisation exists
                  - If Cartesian_visualisation exists, treat as sparse (old format)
                  - Otherwise, default to False (non-sparse)
        """
        try:
            sparse_attr = sync(self._file.get_attr(self._group, 'Sparse'))
            # Handle both boolean and string representations
            if isinstance(sparse_attr, bool):
                return sparse_attr
            elif isinstance(sparse_attr, str):
                return sparse_attr.lower() in ('true', '1', 'yes')
            else:
                return bool(sparse_attr)
        except Exception:
            # If attribute doesn't exist, check for backward compatibility
            # Old files have Cartesian_visualisation which indicates sparse format
            cv_path = concatenate_paths(self._path, brim_obj_names.data.cartesian_visualisation)
            if sync(self._file.object_exists(cv_path)):
                # Old format with Cartesian_visualisation -> sparse
                return True
            else:
                # New format without Cartesian_visualisation -> non-sparse (as per spec default)
                return False
    
    def is_sparse(self) -> bool:
        """
        Returns whether the data is sparse (not on a regular 3D grid).
        
        Returns:
            bool: True if data is sparse, False otherwise.
        """
        return self._is_sparse

    def _load_spatial_mapping(self, load_in_memory: bool=True) -> tuple:
        """
        Load a spatial mapping in the same format as 'Cartesian visualisation',
        irrespectively on whether 'Spatial_map' is defined instead.
        -1 is used for "empty" pixels in the image
        Args:
            load_in_memory (bool): Specify whether the map should be forced to load in memory or just opened as a dataset.
        Returns:
            The spatial map and the corresponding pixel size as a tuple of 3 Metadata.Item, both in the order z, y, x.
        """
        cv = None
        px_size = 3*(Metadata.Item(value=1, units=None),)

        cv_path = concatenate_paths(
            self._path, brim_obj_names.data.cartesian_visualisation)
        sm_path = concatenate_paths(
            self._path, brim_obj_names.data.spatial_map)
        
        # Check if data is non-sparse (on regular 3D grid)
        if not self._is_sparse:
            # For non-sparse data, PSD has shape [z, y, x, spectrum]
            # We need to create a spatial map from the PSD shape
            PSD_path = concatenate_paths(self._path, brim_obj_names.data.PSD)
            if sync(self._file.object_exists(PSD_path)):
                PSD = sync(self._file.open_dataset(PSD_path))
                # For non-sparse data, create identity mapping
                if PSD.ndim >= 3:
                    nZ, nY, nX = PSD.shape[0:3]
                    indices = np.arange(nZ * nY * nX)
                    cv = np.reshape(indices, (nZ, nY, nX))
                    cv = np_array_to_smallest_int_type(cv)
                    
                    # Try to read element_size from Data group level first
                    px_size_val = None
                    px_size_units = None
                    try:
                        px_size_val = sync(self._file.get_attr(self._group, 'element_size'))
                        if px_size_val is not None and len(px_size_val) == 3:
                            px_size_units = sync(units.of_attribute(self._file, self._group, 'element_size'))
                    except Exception:
                        # Fall back to default
                        px_size_val = 3*(1,)
                        warnings.warn("No pixel size defined for non-sparse data")
                    
                    px_size = ()
                    for i in range(3):
                        if isinstance(px_size_val[i], Number):
                            px_size += (Metadata.Item(px_size_val[i], px_size_units), )
                        else:
                            px_size += (Metadata.Item(1, None), )
                    
                    if load_in_memory:
                        cv = np.array(cv)
            
            return cv, px_size
        
        # For sparse data, check Cartesian_visualisation or Spatial_map
        if sync(self._file.object_exists(cv_path)):
            cv = sync(self._file.open_dataset(cv_path))

            #read the pixel size from the 'Cartesian visualisation' dataset
            px_size_val = None
            px_size_units = None
            try:
                px_size_val = sync(self._file.get_attr(cv, 'element_size'))
                if px_size_val is None or len(px_size_val) != 3:
                    raise ValueError(
                        "The 'element_size' attribute of 'Cartesian_visualisation' must be a tuple of 3 elements")
            except Exception:
                px_size_val = 3*(1,)
                warnings.warn(
                    "No pixel size defined for Cartesian visualisation")            
            px_size_units = sync(units.of_attribute(
                    self._file, cv, 'element_size'))
            px_size = ()
            for i in range(3):
                # if px_size_val[i] is not a number, set it to 1 and px_size_units to None
                if isinstance(px_size_val[i], Number):
                    px_size += (Metadata.Item(px_size_val[i], px_size_units), )
                else:
                    px_size += (Metadata.Item(1, None), )
                    

            if load_in_memory:
                cv = np.array(cv)
                cv = np_array_to_smallest_int_type(cv)

        elif sync(self._file.object_exists(sm_path)):
            def load_spatial_map_from_file(self):
                async def load_coordinate_from_sm(coord: str):
                    res = np.empty(0)  # empty array
                    try:
                        res = await self._file.open_dataset(
                            concatenate_paths(sm_path, coord))
                        res = await res.to_np_array()
                        res = np.squeeze(res)  # remove single-dimensional entries
                    except Exception as e:
                        # if the coordinate does not exist, return an empty array
                        pass
                    if len(res.shape) > 1:
                        raise ValueError(
                            f"The 'Spatial_map/{coord}' dataset is not a 1D array as expected")
                    return res

                def check_coord_array(arr, size):
                    if arr.size == 0:
                        return np.zeros(size)
                    elif arr.size != size:
                        raise ValueError(
                            "The 'Spatial_map' dataset is invalid")
                    return arr

                x, y, z = _gather_sync(
                    load_coordinate_from_sm('x'),
                    load_coordinate_from_sm('y'),
                    load_coordinate_from_sm('z')
                    )
                size = max([x.size, y.size, z.size])
                if size == 0:
                    raise ValueError("The 'Spatial_map' dataset is empty")
                x = check_coord_array(x, size)
                y = check_coord_array(y, size)
                z = check_coord_array(z, size)
                return x, y, z

            def calculate_step(x):
                n = len(np.unique(x))
                if n == 1:
                    d = None
                else:
                    d = (np.max(x)-np.min(x))/(n-1)
                return n, d

            x, y, z = load_spatial_map_from_file(self)

            # TODO extend the reconstruction to non-cartesian cases

            nX, dX = calculate_step(x)
            nY, dY = calculate_step(y)
            nZ, dZ = calculate_step(z)

            indices = np_array_to_smallest_int_type(np.lexsort((x, y, z)))
            cv = np.reshape(indices, (nZ, nY, nX))

            px_size_units = sync(units.of_object(self._file, sm_path))
            px_size = ()
            for i in range(3):
                px_sz = (dZ, dY, dX)[i]
                px_unit = px_size_units
                if px_sz is None:
                    px_sz = 1
                    px_unit = None
                px_size += (Metadata.Item(px_sz, px_unit),)

        return cv, px_size

    def get_PSD(self) -> tuple:
        """
        LOW LEVEL FUNCTION

        Retrieve the Power Spectral Density (PSD) and frequency from the current data group.
        Note: this function exposes the internals of the brim file and thus the interface might change in future versions.
        Use only if more specialized functions are not working for your application!
        Returns:
            tuple: (PSD, frequency, PSD_units, frequency_units)
                - PSD: A 2D (or more) numpy array containing all the spectra (see [specs](https://github.com/prevedel-lab/Brillouin-standard-file/blob/main/docs/brim_file_specs.md) for more details).
                - frequency: A numpy array representing the frequency data (see [specs](https://github.com/prevedel-lab/Brillouin-standard-file/blob/main/docs/brim_file_specs.md) for more details).
                - PSD_units: The units of the PSD.
                - frequency_units: The units of the frequency.
        """
        PSD, frequency = _gather_sync(
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.PSD)),
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.frequency))
        )
        # retrieve the units of the PSD and frequency
        PSD_units, frequency_units = _gather_sync(
            units.of_object(self._file, PSD),
            units.of_object(self._file, frequency)
        )

        return PSD, frequency, PSD_units, frequency_units
    
    def get_PSD_as_spatial_map(self) -> tuple:
        """
        Retrieve the Power Spectral Density (PSD) as a spatial map and the frequency from the current data group.
        Returns:
            tuple: (PSD, frequency, PSD_units, frequency_units)
                - PSD: A 4D (or more) numpy array containing all the spectra. Dimensions are z, y, x, [parameters], spectrum.
                - frequency: A numpy array representing the frequency data, which can be broadcastable to PSD.
                - PSD_units: The units of the PSD.
                - frequency_units: The units of the frequency.
        """
        PSD, frequency = _gather_sync(
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.PSD)),        
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.frequency))
            )        
        # retrieve the units of the PSD and frequency
        PSD_units, frequency_units = _gather_sync(
            units.of_object(self._file, PSD),
            units.of_object(self._file, frequency)
        )

        # ensure PSD and frequency are numpy arrays
        PSD = np.array(PSD)  
        frequency = np.array(frequency)  # ensure it's a numpy array
        
        #if the frequency is not the same for all spectra, broadcast it to match the shape of PSD
        if frequency.ndim > 1:
            frequency = np.broadcast_to(frequency, PSD.shape)
        
        # If data is sparse, need to remap using spatial_map
        if self._is_sparse:
            sm = np.array(self._spatial_map)
            # reshape the PSD to have the spatial dimensions first      
            PSD = PSD[sm, ...]
            # reshape the frequency only if it is not the same for all spectra
            if frequency.ndim > 1:
                frequency = frequency[sm, ...]
        # For non-sparse data, PSD is already in the correct shape (z, y, x, spectrum)

        return PSD, frequency, PSD_units, frequency_units

    def get_spectrum(self, index: int) -> tuple:
        """
        Synchronous wrapper for `get_spectrum_async` (see doc for `brimfile.data.Data.get_spectrum_async`)
        """
        return sync(self.get_spectrum_async(index))
    def _linear_index_to_3d(self, index: int, shape: tuple) -> tuple:
        """
        Convert a linear index to 3D coordinates (z, y, x).
        
        Args:
            index (int): Linear index
            shape (tuple): Shape tuple (nZ, nY, nX)
            
        Returns:
            tuple: (z, y, x) coordinates
        """
        nZ, nY, nX = shape
        z = index // (nY * nX)
        remainder = index % (nY * nX)
        y = remainder // nX
        x = remainder % nX
        return (z, y, x)

    async def get_spectrum_async(self, index: int) -> tuple:
        """
        Retrieve a spectrum from the data group.

        Args:
            index (int): The index of the spectrum to retrieve.

        Returns:
            tuple: (PSD, frequency, PSD_units, frequency_units) for the specified index. 
                    PSD can be 1D or more (if there are additional parameters);
                    frequency has the same size as PSD
        Raises:
            IndexError: If the index is out of range for the PSD dataset.
        """
        # index = -1 corresponds to no spectrum
        if index < 0:
            return None, None, None, None
        PSD, frequency = await asyncio.gather(
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.PSD)),                       
            self._file.open_dataset(concatenate_paths(
                self._path, brim_obj_names.data.frequency))
            )
        
        # retrieve the units of the PSD and frequency
        PSD_units, frequency_units = await asyncio.gather(
            units.of_object(self._file, PSD),
            units.of_object(self._file, frequency)
        )
        
        # For sparse data, PSD is 2D [N_points, spectrum], index directly
        # For non-sparse data, PSD is 4D [z, y, x, spectrum], need to convert linear index to 3D
        if self._is_sparse:
            if index >= PSD.shape[0]:
                raise IndexError(
                    f"index {index} out of range for PSD with shape {PSD.shape}")
            
            # map index to the frequency array, considering the broadcasting rules
            index_frequency = (index, ...)
            if frequency.ndim < PSD.ndim:
                # if the frequency has less dimensions than PSD,
                # it's the same for all spatial positions
                index_frequency = (..., )
            #get the spectrum and the corresponding frequency at the specified index
            PSD, frequency = await asyncio.gather(
                _async_getitem(PSD, (index,...)),
                _async_getitem(frequency, index_frequency)
            )
        else:
            # For non-sparse, convert linear index to 3D coordinates
            nZ, nY, nX = PSD.shape[0:3]
            total_points = nZ * nY * nX
            if index >= total_points:
                raise IndexError(
                    f"index {index} out of range for PSD with {total_points} total points")
            
            # Convert linear index to z, y, x coordinates
            z, y, x = self._linear_index_to_3d(index, (nZ, nY, nX))
            
            # map coordinates to the frequency array
            index_frequency = (z, y, x, ...)
            if frequency.ndim == 1:
                # frequency is same for all spatial positions
                index_frequency = (..., )
            elif frequency.ndim < 4:
                # Handle case where frequency has fewer dimensions
                index_frequency = (..., )
            
            #get the spectrum and the corresponding frequency at the specified coordinates
            PSD, frequency = await asyncio.gather(
                _async_getitem(PSD, (z, y, x, ...)),
                _async_getitem(frequency, index_frequency)
            )
        
        #broadcast the frequency to match the shape of PSD if needed
        if frequency.ndim < PSD.ndim:
            frequency = np.broadcast_to(frequency, PSD.shape)
        return PSD, frequency, PSD_units, frequency_units

    def get_spectrum_in_image(self, coor: tuple) -> tuple:
        """
        Retrieve a spectrum from the data group using spatial coordinates.

        Args:
            coor (tuple): A tuple containing the z, y, x coordinates of the spectrum to retrieve.

        Returns:
            tuple: A tuple containing the PSD, frequency, PSD_units, frequency_units for the specified coordinates. See "get_spectrum" for details.
        """
        if len(coor) != 3:
            raise ValueError("coor must contain 3 values for z, y, x")

        index = int(self._spatial_map[coor])
        return self.get_spectrum(index)    
          
    def get_spectrum_and_all_quantities_in_image(self, ar: 'Data.AnalysisResults', coor: tuple, index_peak: int = 0):
        """
            Retrieve the spectrum and all available quantities from the analysis results at a specific spatial coordinate.
            TODO complete the documentation
        """
        if len(coor) != 3:
            raise ValueError("coor must contain 3 values for z, y, x")
        index = int(self._spatial_map[coor])
        spectrum, quantities = _gather_sync(
            self.get_spectrum_async(index),
            ar._get_all_quantities_at_index(index, index_peak)
        )
        return spectrum, quantities

    class AnalysisResults:
        """
        Rapresents the analysis results associated with a Data object.
        """

        class Quantity(Enum):
            """
            Enum representing the type of analysis results.
            """
            Shift = "Shift"
            Width = "Width"
            Amplitude = "Amplitude"
            Offset = "Offset"
            R2 = "R2"
            RMSE = "RMSE"
            Cov_matrix = "Cov_matrix"

        class PeakType(Enum):
            AntiStokes = "AS"
            Stokes = "S"
            average = "avg"
        
        class FitModel(Enum):
            Undefined = "Undefined"
            Lorentzian = "Lorentzian"
            DHO = "DHO"
            Gaussian = "Gaussian"
            Voigt = "Voigt"
            Custom = "Custom"

        def __init__(self, file: FileAbstraction, full_path: str, spatial_map, spatial_map_px_size):
            """
            Initialize the AnalysisResults object.

            Args:
                file (File): The parent File object.
                full_path (str): path of the group storing the analysis results
            """
            self._file = file
            self._path = full_path
            # self._group = file.open_group(full_path)
            self._spatial_map = spatial_map
            self._spatial_map_px_size = spatial_map_px_size

        def get_name(self):
            """
            Returns the name of the Analysis group.
            """
            return sync(get_object_name(self._file, self._path))

        @classmethod
        def _create_new(cls, data: 'Data', index: int) -> 'Data.AnalysisResults':
            """
            Create a new AnalysisResults group.

            Args:
                file (FileAbstraction): The file.
                index (int): The index for the new AnalysisResults group.

            Returns:
                AnalysisResults: The newly created AnalysisResults object.
            """
            group_name = f"{brim_obj_names.data.analysis_results}_{index}"
            ar_full_path = concatenate_paths(data._path, group_name)
            group = sync(data._file.create_group(ar_full_path))
            return cls(data._file, ar_full_path, data._spatial_map, data._spatial_map_px_size)

        def add_data(self, data_AntiStokes=None, data_Stokes=None, fit_model: 'Data.AnalysisResults.FitModel' = None):
            """
            Adds data for the analysis results for AntiStokes and Stokes peaks to the file.
            
            Args:
                data_AntiStokes (dict or list[dict]): A dictionary containing the analysis results for AntiStokes peaks.
                    In case multiple peaks were fitted, it might be a list of dictionaries with each element corresponding to a single peak.
                
                    Each dictionary may include the following keys (plus the corresponding units,  e.g. 'shift_units'):
                        - 'shift': The shift value.
                        - 'width': The width value.
                        - 'amplitude': The amplitude value.
                        - 'offset': The offset value.
                        - 'R2': The R-squared value.
                        - 'RMSE': The root mean square error value.
                        - 'Cov_matrix': The covariance matrix.
                data_Stokes (dict or list[dict]): same as `data_AntiStokes` for the Stokes peaks.
                fit_model (Data.AnalysisResults.FitModel, optional): The fit model used for the analysis. Defaults to None (no attribute is set).

                Both `data_AntiStokes` and `data_Stokes` are optional, but at least one of them must be provided.
            """

            ar_cls = Data.AnalysisResults
            ar_group = sync(self._file.open_group(self._path))

            def add_quantity(qt: Data.AnalysisResults.Quantity, pt: Data.AnalysisResults.PeakType, data, index: int = 0):
                # TODO: check if the data is valid
                sync(self._file.create_dataset(
                    ar_group, ar_cls._get_quantity_name(qt, pt, index), data))

            def add_data_pt(pt: Data.AnalysisResults.PeakType, data, index: int = 0):
                if 'shift' in data:
                    add_quantity(ar_cls.Quantity.Shift,
                                 pt, data['shift'], index)
                    if 'shift_units' in data:
                        self._set_units(data['shift_units'],
                                        ar_cls.Quantity.Shift, pt, index)
                if 'width' in data:
                    add_quantity(ar_cls.Quantity.Width,
                                 pt, data['width'], index)
                    if 'width_units' in data:
                        self._set_units(data['width_units'],
                                        ar_cls.Quantity.Width, pt, index)
                if 'amplitude' in data:
                    add_quantity(ar_cls.Quantity.Amplitude,
                                 pt, data['amplitude'], index)
                    if 'amplitude_units' in data:
                        self._set_units(
                            data['amplitude_units'], ar_cls.Quantity.Amplitude, pt, index)
                if 'offset' in data:
                    add_quantity(ar_cls.Quantity.Offset,
                                 pt, data['offset'], index)
                    if 'offset_units' in data:
                        self._set_units(
                            data['offset_units'], ar_cls.Quantity.Offset, pt, index)
                if 'R2' in data:
                    add_quantity(ar_cls.Quantity.R2, pt, data['R2'], index)
                    if 'R2_units' in data:
                        self._set_units(data['R2_units'],
                                        ar_cls.Quantity.R2, pt, index)
                if 'RMSE' in data:
                    add_quantity(ar_cls.Quantity.RMSE, pt, data['RMSE'], index)
                    if 'RMSE_units' in data:
                        self._set_units(data['RMSE_units'],
                                        ar_cls.Quantity.RMSE, pt, index)
                if 'Cov_matrix' in data:
                    add_quantity(ar_cls.Quantity.Cov_matrix,
                                 pt, data['Cov_matrix'], index)
                    if 'Cov_matrix_units' in data:
                        self._set_units(
                            data['Cov_matrix_units'], ar_cls.Quantity.Cov_matrix, pt, index)

            if data_AntiStokes is not None:
                data_AntiStokes = var_to_singleton(data_AntiStokes)
                for i, d_as in enumerate(data_AntiStokes):
                    add_data_pt(ar_cls.PeakType.AntiStokes, d_as, i)
            if data_Stokes is not None:
                data_Stokes = var_to_singleton(data_Stokes)
                for i, d_s in enumerate(data_Stokes):
                    add_data_pt(ar_cls.PeakType.Stokes, d_s, i)
            if fit_model is not None:
                sync(self._file.create_attr(ar_group, 'Fit_model', fit_model.value))

        def get_units(self, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0) -> str:
            """
            Retrieve the units of a specified quantity from the data file.

            Args:
                qt (Quantity): The quantity for which the units are to be retrieved.
                pt (PeakType, optional): The type of peak (e.g., Stokes or AntiStokes). Defaults to PeakType.AntiStokes.
                index (int, optional): The index of the quantity in case multiple quantities exist. Defaults to 0.

            Returns:
                str: The units of the specified quantity as a string.
            """
            dt_name = Data.AnalysisResults._get_quantity_name(qt, pt, index)
            full_path = concatenate_paths(self._path, dt_name)
            return sync(units.of_object(self._file, full_path))

        def _set_units(self, un: str, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0) -> str:
            """
            Set the units of a specified quantity.

            Args:
                un (str): The units to be set.
                qt (Quantity): The quantity for which the units are to be set.
                pt (PeakType, optional): The type of peak (e.g., Stokes or AntiStokes). Defaults to PeakType.AntiStokes.
                index (int, optional): The index of the quantity in case multiple quantities exist. Defaults to 0.

            Returns:
                str: The units of the specified quantity as a string.
            """
            dt_name = Data.AnalysisResults._get_quantity_name(qt, pt, index)
            full_path = concatenate_paths(self._path, dt_name)
            return units.add_to_object(self._file, full_path, un)
        
        @property
        def fit_model(self) -> 'Data.AnalysisResults.FitModel':
            """
            Retrieve the fit model used for the analysis.

            Returns:
                Data.AnalysisResults.FitModel: The fit model used for the analysis.
            """
            if not hasattr(self, '_fit_model'):
                try:
                    fit_model_str = sync(self._file.get_attr(self._path, 'Fit_model'))
                    self._fit_model = Data.AnalysisResults.FitModel(fit_model_str)
                except Exception as e:
                    if isinstance(e, ValueError):
                        warnings.warn(
                            f"Unknown fit model '{fit_model_str}' found in the file.")
                    self._fit_model = Data.AnalysisResults.FitModel.Undefined        
            return self._fit_model

        def save_image_to_OMETiff(self, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0, filename: str = None) -> str:
            """
            Saves the image corresponding to the specified quantity and index to an OMETiff file.

            Args:
                qt (Quantity): The quantity to retrieve the image for (e.g. shift).
                pt (PeakType, optional): The type of peak to consider (default is PeakType.AntiStokes).
                index (int, optional): The index of the data to retrieve, if multiple are present (default is 0).
                filename (str, optional): The name of the file to save the image to. If None, a default name will be used.

            Returns:
                str: The path to the saved OMETiff file.
            """
            try:
                import tifffile
            except ImportError:
                raise ModuleNotFoundError(
                    "The tifffile module is required for saving to OME-Tiff. Please install it using 'pip install tifffile'.")
            
            if filename is None:
                filename = f"{qt.value}_{pt.value}_{index}.ome.tif"
            if not filename.endswith('.ome.tif'):
                filename += '.ome.tif'
            img, px_size = self.get_image(qt, pt, index)
            if img.ndim > 3:
                raise NotImplementedError(
                    "Saving images with more than 3 dimensions is not supported yet.")
            with tifffile.TiffWriter(filename, bigtiff=True) as tif:
                metadata = {
                    'axes': 'ZYX',
                    'PhysicalSizeX': px_size[2].value,
                    'PhysicalSizeXUnit': px_size[2].units,
                    'PhysicalSizeY': px_size[1].value,
                    'PhysicalSizeYUnit': px_size[1].units,
                    'PhysicalSizeZ': px_size[0].value,
                    'PhysicalSizeZUnit': px_size[0].units,
                }
                tif.write(img, metadata=metadata)
            return filename

        def get_image(self, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0) -> tuple:
            """
            Retrieves an image (spatial map) based on the specified quantity, peak type, and index.

            Args:
                qt (Quantity): The quantity to retrieve the image for (e.g. shift).
                pt (PeakType, optional): The type of peak to consider (default is PeakType.AntiStokes).
                index (int, optional): The index of the data to retrieve, if multiple are present (default is 0).

            Returns:
                A tuple containing the image corresponding to the specified quantity and index and the corresponding pixel size.
                The image is a 3D dataset where the dimensions are z, y, x.
                If there are additional parameters, more dimensions are added in the order z, y, x, par1, par2, ...
                The pixel size is a tuple of 3 Metadata.Item in the order z, y, x.
            """
            pt_type = Data.AnalysisResults.PeakType
            data = None
            if pt == pt_type.average:
                peaks = self.list_existing_peak_types(index)
                match len(peaks):
                    case 0:
                        raise ValueError(
                            "No peaks found for the specified index. Cannot compute average.")
                    case 1:
                        data = np.array(sync(self._get_quantity(qt, peaks[0], index)))
                    case 2:
                        data1, data2 = _gather_sync(
                            self._get_quantity(qt, peaks[0], index),
                            self._get_quantity(qt, peaks[1], index)
                            )
                        data = (np.abs(data1) + np.abs(data2))/2
            else:
                data = np.array(sync(self._get_quantity(qt, pt, index)))
            sm = np.array(self._spatial_map)
            img = data[sm, ...]
            img[sm<0, ...] = np.nan  # set invalid pixels to NaN
            return img, self._spatial_map_px_size
        def get_quantity_at_pixel(self, coord: tuple, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0):
            """
            Synchronous wrapper for `get_quantity_at_pixel_async` (see doc for `brimfile.data.Data.AnalysisResults.get_quantity_at_pixel_async`)
            """
            return sync(self.get_quantity_at_pixel_async(coord, qt, pt, index))
        async def get_quantity_at_pixel_async(self, coord: tuple, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0):
            """
            Retrieves the specified quantity in the image at coord, based on the peak type and index.

            Args:
                coord (tuple): A tuple of 3 elements corresponding to the z, y, x coordinate in the image
                qt (Quantity): The quantity to retrieve the image for (e.g. shift).
                pt (PeakType, optional): The type of peak to consider (default is PeakType.AntiStokes).
                index (int, optional): The index of the data to retrieve, if multiple peaks are present (default is 0).

            Returns:
                The requested quantity, which is a scalar or a multidimensional array (depending on whether there are additional parameters in the current Data group)
            """
            if len(coord) != 3:
                raise ValueError(
                    "'coord' must have 3 elements corresponding to z, y, x")
            i = self._spatial_map[*coord]
            assert i.size == 1
            if i<0:
                return np.nan  # invalid pixel
            i = int(i)

            pt_type = Data.AnalysisResults.PeakType
            value = None
            if pt == pt_type.average:
                value = None
                peaks = await self.list_existing_peak_types_async(index)
                match len(peaks):
                    case 0:
                        raise ValueError(
                            "No peaks found for the specified index. Cannot compute average.")
                    case 1:
                        data = await self._get_quantity(qt, peaks[0], index)
                        value = await _async_getitem(data, (i, ...))
                    case 2:
                        data_p0, data_p1 = await asyncio.gather(
                            self._get_quantity(qt, peaks[0], index),
                            self._get_quantity(qt, peaks[1], index)
                        )
                        value1, value2 = await asyncio.gather(
                            _async_getitem(data_p0, (i, ...)),
                            _async_getitem(data_p1, (i, ...))
                        )
                        value = (np.abs(value1) + np.abs(value2))/2
            else:
                data = await self._get_quantity(qt, pt, index)
                value = await _async_getitem(data, (i, ...))
            return value
        def get_all_quantities_in_image(self, coor: tuple, index_peak: int = 0) -> dict:
            """
            Retrieve all available quantities at a specific spatial coordinate.
            see `brimfile.data.Data.AnalysisResults._get_all_quantities_at_index` for more details
            TODO complete the documentation
            """
            if len(coor) != 3:
                raise ValueError("coor must contain 3 values for z, y, x")
            index = int(self._spatial_map[coor])
            return sync(self._get_all_quantities_at_index(index, index_peak))
        async def _get_all_quantities_at_index(self, index: int, index_peak: int = 0) -> dict:
            """
            Retrieve all available quantities for a specific spatial index.
            Args:
                index (int): The spatial index to retrieve quantities for.
                index_peak (int, optional): The index of the data to retrieve, if multiple peaks are present (default is 0).
            Returns:
                dict: A dictionary of Metadata.Item in the form `result[quantity.name][peak.name] = bls.Metadata.Item(value, units)`
            """
            async def _get_existing_quantity_at_index_async(self,  pt: Data.AnalysisResults.PeakType = Data.AnalysisResults.PeakType.AntiStokes):
                as_cls = Data.AnalysisResults
                qts_ls = ()
                dts_ls = ()

                qts = [qt for qt in as_cls.Quantity]
                coros = [self._file.open_dataset(concatenate_paths(self._path, as_cls._get_quantity_name(qt, pt, index_peak))) for qt in qts]
                
                # open the datasets asynchronously, excluding those that do not exist
                opened_dts = await asyncio.gather(*coros, return_exceptions=True)
                for i, opened_qt in enumerate(opened_dts):
                    if not isinstance(opened_qt, Exception):
                        qts_ls += (qts[i],)
                        dts_ls += (opened_dts[i],)
                # get the values at the specified index
                coros_values = [_async_getitem(dt, (index, ...)) for dt in dts_ls]
                coros_units = [units.of_object(self._file, dt) for dt in dts_ls]
                ret_ls = await asyncio.gather(*coros_values, *coros_units)
                n = len(coros_values)
                value_ls = [Metadata.Item(ret_ls[i], ret_ls[n+i]) for i in range(n)]
                return qts_ls, value_ls
            antiStokes, stokes = await asyncio.gather(
                _get_existing_quantity_at_index_async(self, Data.AnalysisResults.PeakType.AntiStokes),
                _get_existing_quantity_at_index_async(self, Data.AnalysisResults.PeakType.Stokes)
            )
            res = {}
            # combine the results, including the average
            for qt in (set(antiStokes[0]) | set(stokes[0])):
                res[qt.name] = {}
                pts = ()
                #Stokes
                if qt in stokes[0]:
                    res[qt.name][Data.AnalysisResults.PeakType.Stokes.name] = stokes[1][stokes[0].index(qt)]
                    pts += (Data.AnalysisResults.PeakType.Stokes,)
                #AntiStokes
                if qt in antiStokes[0]:
                    res[qt.name][Data.AnalysisResults.PeakType.AntiStokes.name] = antiStokes[1][antiStokes[0].index(qt)]
                    pts += (Data.AnalysisResults.PeakType.AntiStokes,)
                #average getting the units of the first peak
                res[qt.name][Data.AnalysisResults.PeakType.average.name] = Metadata.Item(
                    np.mean([np.abs(res[qt.name][pt.name].value) for pt in pts]), 
                    res[qt.name][pts[0].name].units
                    )
                if not all(res[qt.name][pt.name].units == res[qt.name][pts[0].name].units for pt in pts):
                    warnings.warn(f"The units of {pts} are not consistent.")
            return res

        @classmethod
        def _get_quantity_name(cls, qt: Quantity, pt: PeakType, index: int) -> str:
            """
            Returns the name of the dataset correponding to the specific Quantity, PeakType and index

            Args:
                qt (Quantity)   
                pt (PeakType)  
                intex (int): in case of multiple peaks fitted, the index of the peak to consider       
            """
            if not pt in (cls.PeakType.AntiStokes, cls.PeakType.Stokes):
                raise ValueError("pt has to be either Stokes or AntiStokes")
            if qt == cls.Quantity.R2 or qt == cls.Quantity.RMSE or qt == cls.Quantity.Cov_matrix:
                name = f"Fit_error_{str(pt.value)}_{index}/{str(qt.value)}"
            else:
                name = f"{str(qt.value)}_{str(pt.value)}_{index}"
            return name

        async def _get_quantity(self, qt: Quantity, pt: PeakType = PeakType.AntiStokes, index: int = 0):
            """
            Retrieve a specific quantity dataset from the file.

            Args:
                qt (Quantity): The type of quantity to retrieve.
                pt (PeakType, optional): The peak type to consider (default is PeakType.AntiStokes).
                index (int, optional): The index of the quantity if multiple peaks are available (default is 0).

            Returns:
                The dataset corresponding to the specified quantity, as stored in the file.

            """

            dt_name = Data.AnalysisResults._get_quantity_name(qt, pt, index)
            full_path = concatenate_paths(self._path, dt_name)
            return await self._file.open_dataset(full_path)

        def list_existing_peak_types(self, index: int = 0) -> tuple:
            """
            Synchronous wrapper for `list_existing_peak_types_async` (see doc for `brimfile.data.Data.AnalysisResults.list_existing_peak_types_async`)
            """
            return sync(self.list_existing_peak_types_async(index)) 
        async def list_existing_peak_types_async(self, index: int = 0) -> tuple:
            """
            Returns a tuple of existing peak types (Stokes and/or AntiStokes) for the specified index.
            Args:
                index (int, optional): The index of the peak to check (in case of multi-peak fit). Defaults to 0.
            Returns:
                tuple: A tuple containing `PeakType` members (`Stokes`, `AntiStokes`) that exist for the given index.
            """

            as_cls = Data.AnalysisResults
            shift_s_name = as_cls._get_quantity_name(
                as_cls.Quantity.Shift, as_cls.PeakType.Stokes, index)
            shift_as_name = as_cls._get_quantity_name(
                as_cls.Quantity.Shift, as_cls.PeakType.AntiStokes, index)
            ls = ()
            coro_as_exists = self._file.object_exists(concatenate_paths(self._path, shift_as_name))
            coro_s_exists = self._file.object_exists(concatenate_paths(self._path, shift_s_name))
            as_exists, s_exists = await asyncio.gather(coro_as_exists, coro_s_exists)
            if as_exists:
                ls += (as_cls.PeakType.AntiStokes,)
            if s_exists:
                ls += (as_cls.PeakType.Stokes,)
            return ls

        def list_existing_quantities(self,  pt: PeakType = PeakType.AntiStokes, index: int = 0) -> tuple:
            """
            Synchronous wrapper for `list_existing_quantities_async` (see doc for `brimfile.data.Data.AnalysisResults.list_existing_quantities_async`)
            """
            return sync(self.list_existing_quantities_async(pt, index))
        async def list_existing_quantities_async(self,  pt: PeakType = PeakType.AntiStokes, index: int = 0) -> tuple:
            """
            Returns a tuple of existing quantities for the specified index.
            Args:
                index (int, optional): The index of the peak to check (in case of multi-peak fit). Defaults to 0.
            Returns:
                tuple: A tuple containing `Quantity` members that exist for the given index.
            """
            as_cls = Data.AnalysisResults
            ls = ()

            qts = [qt for qt in as_cls.Quantity]
            coros = [self._file.object_exists(concatenate_paths(self._path, as_cls._get_quantity_name(qt, pt, index))) for qt in qts]
            
            qt_exists = await asyncio.gather(*coros)
            for i, exists in enumerate(qt_exists):
                if exists:
                    ls += (qts[i],)
            return ls

    def get_metadata(self):
        """
        Returns the metadata associated with the current Data group
        Note that this contains both the general metadata stored in the file (which might be redifined by the specific data group)
        and the ones specific for this data group
        """
        return Metadata(self._file, self._path)

    def get_num_parameters(self) -> tuple:
        """
        Retrieves the number of parameters

        Returns:
            tuple: The shape of the parameters if they exist, otherwise an empty tuple.
        """
        pars, _ = self.get_parameters()
        return pars.shape if pars is not None else ()

    def get_parameters(self) -> list:
        """
        Retrieves the parameters  and their associated names.

        If PSD.ndims > 2, the parameters are stored in a separate dataset.

        Returns:
            list: A tuple containing the parameters and their names if there are any, otherwise None.
        """
        pars_full_path = concatenate_paths(
            self._path, brim_obj_names.data.parameters)
        if sync(self._file.object_exists(pars_full_path)):
            pars = sync(self._file.open_dataset(pars_full_path))
            pars_names = sync(self._file.get_attr(pars, 'Name'))
            return (pars, pars_names)
        return (None, None)

    def create_analysis_results_group(self, data_AntiStokes, data_Stokes=None, index: int = None, name: str = None, fit_model: 'Data.AnalysisResults.FitModel' = None) -> AnalysisResults:
        """
        Adds a new AnalysisResults entry to the current data group.
        Parameters:
            data_AntiStokes (dict or list[dict]): contains the same elements as the ones in `AnalysisResults.add_data`,
                but all the quantities (i.d. 'shift', 'width', etc.) are 3D, corresponding to the spatial positions (z, y, x).
            data_Stokes (dict or list[dict]): same as data_AntiStokes for the Stokes peaks.
            index (int, optional): The index for the new data entry. If None, the next available index is used. Defaults to None.
            name (str, optional): The name for the new Analysis group. Defaults to None.
            fit_model (Data.AnalysisResults.FitModel, optional): The fit model used for the analysis. Defaults to None (no attribute is set).
        Returns:
            AnalysisResults: The newly created AnalysisResults object.
        Raises:
            IndexError: If the specified index already exists in the dataset.
            ValueError: If any of the data provided is not valid or consistent
        """
        def flatten_data(data: dict):
            if data is None:
                return None
            data = var_to_singleton(data)
            out_data = []
            for dn in data:
                for k in dn.keys():
                    if not k.endswith('_units'):
                        d = dn[k]
                        # For non-sparse data, data arrays should have shape [z, y, x]
                        # For sparse data (old format), data arrays can be 3D (need flattening) or 1D (already flat)
                        if not self._is_sparse:
                            if d.ndim != 3 or d.shape != self._spatial_map.shape:
                                raise ValueError(
                                    f"'{k}' must have 3 dimensions (z, y, x) and same shape as the spatial map ({self._spatial_map.shape})")
                            dn[k] = np.reshape(d, -1)  # flatten the data
                        else:
                            # For sparse data, accept both 3D (old API) and 1D (new API)
                            if d.ndim == 3:
                                # Old API: 3D array, need to flatten
                                if d.shape != self._spatial_map.shape:
                                    raise ValueError(
                                        f"'{k}' must have same shape as the spatial map ({self._spatial_map.shape}) when 3D")
                                dn[k] = np.reshape(d, -1)  # flatten the data
                            elif d.ndim == 1:
                                # New API: already 1D, just verify length
                                expected_length = np.prod(self._spatial_map.shape)
                                if d.shape[0] != expected_length:
                                    raise ValueError(
                                        f"'{k}' must have length {expected_length} to match spatial map")
                            else:
                                raise ValueError(
                                    f"'{k}' must be either 3D or 1D for sparse data")
                out_data.append(dn)
            return out_data
        data_AntiStokes = flatten_data(data_AntiStokes)
        data_Stokes = flatten_data(data_Stokes)
        return self.create_analysis_results_group_raw(data_AntiStokes, data_Stokes, index, name, fit_model=fit_model)

    def create_analysis_results_group_raw(self, data_AntiStokes, data_Stokes=None, index: int = None, name: str = None, fit_model: 'Data.AnalysisResults.FitModel' = None) -> AnalysisResults:
        """
        Adds a new AnalysisResults entry to the current data group.
        Parameters:
            data_AntiStokes (dict or list[dict]): see documentation for AnalysisResults.add_data
            data_Stokes (dict or list[dict]): same as data_AntiStokes for the Stokes peaks.
            index (int, optional): The index for the new data entry. If None, the next available index is used. Defaults to None.
            name (str, optional): The name for the new Analysis group. Defaults to None.
            fit_model (Data.AnalysisResults.FitModel, optional): The fit model used for the analysis. Defaults to None (no attribute is set).
        Returns:
            AnalysisResults: The newly created AnalysisResults object.
        Raises:
            IndexError: If the specified index already exists in the dataset.
            ValueError: If any of the data provided is not valid or consistent
        """
        if index is not None:
            try:
                self.get_analysis_results(index)
            except IndexError:
                pass
            else:
                # If the group already exists, raise an error
                raise IndexError(
                    f"Analysis {index} already exists in {self._path}")
        else:
            ar_groups = self.list_AnalysisResults()
            indices = [ar['index'] for ar in ar_groups]
            indices.sort()
            index = indices[-1] + 1 if indices else 0  # Next available index

        ar = Data.AnalysisResults._create_new(self, index)
        if name is not None:
            set_object_name(self._file, ar._path, name)
        ar.add_data(data_AntiStokes, data_Stokes, fit_model=fit_model)

        return ar

    def list_AnalysisResults(self, retrieve_custom_name=False) -> list:
        """
        List all AnalysisResults groups in the current data group. The list is ordered by index.

        Returns:
            list: A list of dictionaries, each containing:
                - 'name' (str): The name of the AnalysisResults group.
                - 'index' (int): The index extracted from the group name.
                - 'custom_name' (str, optional): if retrieve_custom_name==True, it contains the name of the AnalysisResults group as returned from utils.get_object_name.
        """

        analysis_results_groups = []

        matched_objs = list_objects_matching_pattern(
            self._file, self._group, brim_obj_names.data.analysis_results + r"_(\d+)$")
        async def _make_dict_item(matched_obj, retrieve_custom_name):
            name = matched_obj[0]
            index = int(matched_obj[1])
            curr_obj_dict = {'name': name, 'index': index}
            if retrieve_custom_name:
                ar_path = concatenate_paths(self._path, name)
                custom_name = await get_object_name(self._file, ar_path)
                curr_obj_dict['custom_name'] = custom_name
            return curr_obj_dict
        coros = [_make_dict_item(matched_obj, retrieve_custom_name) for matched_obj in matched_objs]
        dicts = _gather_sync(*coros)
        for dict_item in dicts:
            analysis_results_groups.append(dict_item)
        # Sort the data groups by index
        analysis_results_groups.sort(key=lambda x: x['index'])

        return analysis_results_groups

    def get_analysis_results(self, index: int = 0) -> AnalysisResults:
        """
        Returns the AnalysisResults at the specified index

        Args:
            index (int)                

        Raises:
            IndexError: If there is no analysis with the corresponding index
        """
        name = None
        ls = self.list_AnalysisResults()
        for el in ls:
            if el['index'] == index:
                name = el['name']
                break
        if name is None:
            raise IndexError(f"Analysis {index} not found")
        path = concatenate_paths(self._path, name)
        return Data.AnalysisResults(self._file, path, self._spatial_map, self._spatial_map_px_size)

    def add_data(self, PSD: np.ndarray, frequency: np.ndarray, scanning: dict, freq_units='GHz', 
                 element_size: tuple = None, element_size_unit: str = None,
                 timestamp: np.ndarray = None, compression: FileAbstraction.Compression = FileAbstraction.Compression(),
                 sparse: bool = None):
        """
        Add data to the current data group.

        This method adds the provided PSD, frequency, and scanning data to the HDF5 group 
        associated with this `Data` object. It validates the inputs to ensure they meet 
        the required specifications before adding them.

        Args:
            PSD (np.ndarray): A numpy array representing the Power Spectral Density (PSD) data. 
                The last dimension contains the spectra.
                For sparse data: 2D array with dimensions [N_points, spectrum]
                For non-sparse data: 4D array with dimensions [z, y, x, spectrum]
            frequency (np.ndarray): A 1D or multi-dimensional numpy array representing the frequency data. 
                It must be broadcastable to the shape of the PSD array.
            scanning (dict): A dictionary containing scanning-related data. It may include:
                - 'Spatial_map' (optional): A dictionary containing (up to) 3 arrays (x, y, z) and a string (units)
                - 'Cartesian_visualisation' (optional): A 3D numpy array containing the association between spatial position and spectra.
                   It must have integer values between 0 and PSD.shape[0]-1, or -1 for invalid entries.
                - 'Cartesian_visualisation_pixel' (optional): A list or array of 3 float values 
                  representing the pixel size in the z, y, and x dimensions (unused dimensions can be set to None).
                - 'Cartesian_visualisation_pixel_unit' (optional): A string representing the unit of the pixel size (e.g. 'um').
            element_size (tuple, optional): Pixel size for z, y, x dimensions. Used for non-sparse data at Data group level.
            element_size_unit (str, optional): Unit for element_size.
            timestamp (np.ndarray): the timestamp associated with each spectrum.
                It must be a 1D array with the same length as the PSD array.
            sparse (bool, optional): If True, sets Sparse attribute to true. If False, sets to false. 
                If None, doesn't set the attribute (defaults to false).


        Raises:
            ValueError: If any of the data provided is not valid or consistent
        """

        # Check if frequency is broadcastable to PSD
        try:
            np.broadcast_shapes(tuple(frequency.shape), tuple(PSD.shape))
        except ValueError as e:
            raise ValueError(f"frequency (shape: {frequency.shape}) is not broadcastable to PSD (shape: {PSD.shape}): {e}")

        # Set sparse attribute if specified
        if sparse is not None:
            sync(self._file.create_attr(self._group, 'Sparse', sparse))
            # Update internal state
            self._is_sparse = sparse
        
        # Add element_size at Data group level if provided
        if element_size is not None:
            sync(self._file.create_attr(self._group, 'element_size', element_size))
            if element_size_unit is not None:
                units.add_to_attribute(self._file, self._group, 'element_size', element_size_unit)

        # define the scanning_is_valid variable to check if at least one of 'Spatial_map' or 'Cartesian_visualisation'
        # is present in the scanning dictionary (or if non-sparse data with no scanning info)
        scanning_is_valid = False
        
        # For non-sparse data, scanning is optional
        if not self._is_sparse and not scanning:
            scanning_is_valid = True
            
        if 'Spatial_map' in scanning:
            sm = scanning['Spatial_map']
            size = 0

            def check_coor(coor: str):
                if coor in sm:
                    sm[coor] = np.array(sm[coor])
                    size1 = sm[coor].size
                    if size1 != size and size != 0:
                        raise ValueError(
                            f"'{coor}' in 'Spatial_map' is invalid!")
                    return size1
            size = check_coor('x')
            size = check_coor('y')
            size = check_coor('z')
            if size == 0:
                raise ValueError(
                    "'Spatial_map' should contain at least one x, y or z")
            scanning_is_valid = True
        if 'Cartesian_visualisation' in scanning:
            cv = scanning['Cartesian_visualisation']
            if not isinstance(cv, np.ndarray) or cv.ndim != 3:
                raise ValueError(
                    "Cartesian_visualisation must be a 3D numpy array")
            if not np.issubdtype(cv.dtype, np.integer) or np.min(cv) < -1 or np.max(cv) >= PSD.shape[0]:
                raise ValueError(
                    "Cartesian_visualisation values must be integers between -1 and PSD.shape[0]-1")
            if 'Cartesian_visualisation_pixel' in scanning:
                if len(scanning['Cartesian_visualisation_pixel']) != 3:
                    raise ValueError(
                        "Cartesian_visualisation_pixel must always contain 3 values for z, y, x (set to None if not used)")
            else:
                warnings.warn(
                    "It is recommended to add 'Cartesian_visualisation_pixel' to the scanning dictionary, to define the pixel size")
            scanning_is_valid = True
        if not scanning_is_valid:
            raise ValueError("scanning is not valid")

        if timestamp is not None:
            if not isinstance(timestamp, np.ndarray) or timestamp.ndim != 1 or len(timestamp) != PSD.shape[0]:
                raise ValueError("timestamp is not compatible with PSD")

        # TODO: add and validate additional datasets (i.e. 'Parameters', 'Calibration_index', etc.)

        # Add datasets to the group
        def determine_chunk_size(arr: np.array) -> tuple:
            """"
            Use the same heuristic as the zarr library to determine the chunk size, but without splitting the last dimension
            """
            shape = arr.shape
            typesize = arr.itemsize
            #if the array is 1D, do not chunk it
            if len(shape) <= 1:
                return (shape[0],)
            target_sizes = _guess_chunks.__kwdefaults__
            # divide the target size by the last dimension size to get the chunk size for the other dimensions
            target_sizes = {k: target_sizes[k] // shape[-1] 
                            for k in target_sizes.keys()}
            chunks = _guess_chunks(shape[0:-1], typesize, arr.nbytes, **target_sizes)
            return chunks + (shape[-1],)  # keep the last dimension size unchanged
        sync(self._file.create_dataset(
            self._group, brim_obj_names.data.PSD, data=PSD,
            chunk_size=determine_chunk_size(PSD), compression=compression))
        freq_ds = sync(self._file.create_dataset(
            self._group,  brim_obj_names.data.frequency, data=frequency,
            chunk_size=determine_chunk_size(frequency), compression=compression))
        units.add_to_object(self._file, freq_ds, freq_units)

        if 'Spatial_map' in scanning:
            sm = scanning['Spatial_map']
            sm_group = sync(self._file.create_group(concatenate_paths(
                self._path, brim_obj_names.data.spatial_map)))
            if 'units' in sm:
                units.add_to_object(self._file, sm_group, sm['units'])

            def add_sm_dataset(coord: str):
                if coord in sm:
                    coord_dts = sync(self._file.create_dataset(
                        sm_group, coord, data=sm[coord], compression=compression))

            add_sm_dataset('x')
            add_sm_dataset('y')
            add_sm_dataset('z')
        if 'Cartesian_visualisation' in scanning:
            # convert the Cartesian_visualisation to the smallest integer type
            cv_arr = np_array_to_smallest_int_type(scanning['Cartesian_visualisation'])
            cv = sync(self._file.create_dataset(self._group, brim_obj_names.data.cartesian_visualisation,
                                           data=cv_arr, compression=compression))
            if 'Cartesian_visualisation_pixel' in scanning:
                sync(self._file.create_attr(
                    cv, 'element_size', scanning['Cartesian_visualisation_pixel']))
                if 'Cartesian_visualisation_pixel_unit' in scanning:
                    px_unit = scanning['Cartesian_visualisation_pixel_unit']
                else:
                    warnings.warn(
                        "No unit provided for Cartesian_visualisation_pixel, defaulting to 'um'")
                    px_unit = 'um'
                units.add_to_attribute(self._file, cv, 'element_size', px_unit)

        self._spatial_map, self._spatial_map_px_size = self._load_spatial_mapping()

        if timestamp is not None:
            sync(self._file.create_dataset(
                self._group, 'Timestamp', data=timestamp, compression=compression))

    @staticmethod
    def list_data_groups(file: FileAbstraction, retrieve_custom_name=False) -> list:
        """
        List all data groups in the brim file. The list is ordered by index.

        Returns:
            list: A list of dictionaries, each containing:
                - 'name' (str): The name of the data group in the file.
                - 'index' (int): The index extracted from the group name.
                - 'custom_name' (str, optional): if retrieve_custom_name==True, it contains the name of the data group as returned from utils.get_object_name.
        """

        data_groups = []

        matched_objs = list_objects_matching_pattern(
            file, brim_obj_names.Brillouin_base_path, brim_obj_names.data.base_group + r"_(\d+)$")
        
        async def _make_dict_item(matched_obj, retrieve_custom_name):
            name = matched_obj[0]
            index = int(matched_obj[1])
            curr_obj_dict = {'name': name, 'index': index}
            if retrieve_custom_name:
                path = concatenate_paths(
                    brim_obj_names.Brillouin_base_path, name)
                custom_name = await get_object_name(file, path)
                curr_obj_dict['custom_name'] = custom_name
            return curr_obj_dict
        
        coros = [_make_dict_item(matched_obj, retrieve_custom_name) for matched_obj in matched_objs]
        dicts = _gather_sync(*coros)
        for dict_item in dicts:
            data_groups.append(dict_item)        
        # Sort the data groups by index
        data_groups.sort(key=lambda x: x['index'])

        return data_groups

    @staticmethod
    def _get_existing_group_name(file: FileAbstraction, index: int) -> str:
        """
        Get the name of an existing data group by index.

        Args:
            file (File): The parent File object.
            index (int): The index of the data group.

        Returns:
            str: The name of the data group, or None if not found.
        """
        group_name: str = None
        data_groups = Data.list_data_groups(file)
        for dg in data_groups:
            if dg['index'] == index:
                group_name = dg['name']
                break
        return group_name

    @classmethod
    def _create_new(cls, file: FileAbstraction, index: int, name: str = None) -> 'Data':
        """
        Create a new data group with the specified index.

        Args:
            file (File): The parent File object.
            index (int): The index for the new data group.
            name (str, optional): The name for the new data group. Defaults to None.

        Returns:
            Data: The newly created Data object.
        """
        group_name = Data._generate_group_name(index)
        group = sync(file.create_group(concatenate_paths(
            brim_obj_names.Brillouin_base_path, group_name)))
        if name is not None:
            set_object_name(file, group, name)
        return cls(file, concatenate_paths(brim_obj_names.Brillouin_base_path, group_name))

    @staticmethod
    def _generate_group_name(index: int, n_digits: int = None) -> str:
        """
        Generate a name for a data group based on the index.

        Args:
            index (int): The index for the data group.
            n_digits (int, optional): The number of digits to pad the index with. If None no padding is applied. Defaults to None.

        Returns:
            str: The generated group name.

        Raises:
            ValueError: If the index is negative.
        """
        if index < 0:
            raise ValueError("index must be positive")
        num = str(index)
        if n_digits is not None:
            num = num.zfill(n_digits)
        return f"{brim_obj_names.data.base_group}_{num}"
