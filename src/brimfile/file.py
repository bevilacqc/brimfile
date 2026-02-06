import numpy as np
import warnings

from .data import Data
from .metadata import Metadata

from .utils import concatenate_paths
from .constants import brim_obj_names

from .file_abstraction import FileAbstraction, StoreType, sync

# don't import _AbstractFile if running in pyodide (it is defined in js)
import sys
if "pyodide" not in sys.modules:
    from .file_abstraction import _AbstractFile

__docformat__ = "google"

class File:
    """
    Represents a brim file with Brillouin data, extending h5py.File.
    """

    if "pyodide" in sys.modules:
        def __init__(self, file):
            self._file = file
            if not self.is_valid():
                raise ValueError("The brim file is not valid!")
    else:
        def __init__(self, filename: str, mode: str = 'r', store_type: StoreType = StoreType.AUTO):
            """
            Initialize the File object.

            Args:
                filename (str): Path to the brim file.
                mode: {'r', 'r+', 'a', 'w', 'w-'} the mode for opening the file (default is 'r' for read-only).
                            See the definition of `mode` in `brimfile.file_abstraction._zarrFile.__init__()` for more details.
                            'r' means read only (must exist); 'r+' means read/write (must exist);
                            'a' means read/write (create if doesn't exist); 'w' means create (overwrite if exists); 'w-' means create (fail if exists).
                store_type (StoreType): Type of the store to use, as defined in `brimfile.file_abstraction.StoreType`. Default is 'AUTO'.
            """
            self._file = _AbstractFile(
                filename, mode=mode, store_type=store_type)
            if not self.is_valid():
                raise ValueError("The brim file is not valid!")
            
    def __del__(self):
        try:
            self.close()
        except Exception as e:            
            # don't throw an error if the file cannot be closed
            warnings.warn(f"Cannot close the file: {e}")

    def close(self) -> None:
        self._file.close()

    def is_read_only(self) -> bool:
        return sync(self._file.is_read_only())

    def is_valid(self) -> bool:
        """
        Check if the file is a valid brim file.

        Returns:
            bool: True if the file is valid, False otherwise.
        """
        # TODO validate file against https://github.com/prevedel-lab/Brillouin-standard-file/blob/main/docs/brim_file_specs.md
        return True

    @classmethod
    def create(cls, filename: str, store_type: StoreType = StoreType.AUTO, brim_version: str = '0.1') -> 'File':
        """
        Create a new brim file with the specified filename. If the file exists already it will generate an error.

        Args:
            filename (str): Path to the brim file to be created.
            store_type (StoreType): Type of the store to use, as defined in `brimfile.file_abstraction.StoreType`. Default is 'AUTO'.
            brim_version (str): Version of the brim file format to use. Default is '0.1'.

        Returns:
            File: An instance of the File class representing the newly created brim file.
            store_type (str): Type of the store to use, as defined in `brimfile.file_abstraction.StoreType`. Default is 'AUTO'.
        """
        f = cls(filename, mode='w-', store_type=store_type)

        # File version
        sync(f._file.create_attr('/', 'brim_version', brim_version))

        # Root Brillouin_data group
        fr = sync(f._file.create_group(brim_obj_names.Brillouin_base_path))

        return f

    def create_data_group(self, PSD: np.ndarray, frequency: np.ndarray, px_size_um: tuple, index: int = None, name: str = None, compression: FileAbstraction.Compression = FileAbstraction.Compression(), sparse: bool = None) -> 'Data':
        """
        Adds a new data entry to the file.
        Parameters:
            PSD (np.ndarray): The Power Spectral Density (PSD) data to be added. 
                Must be 4D with dimensions z, y, x, spectrum (will be flattened if sparse is not False)
            frequency (np.ndarray): The frequency data corresponding to the PSD. 
                Must be 4D or 1D (in which case the frequency axis is assumed the same for all the spatial coordinates)
            px_size_um (tuple): A tuple of 3 elements, in the order z,y,x, corresponding to the pixel size in um. Unused dimensions can be set to None.
            index (int, optional): The index for the new data group. If None, the next available index is used. Defaults to None.
            name (str, optional): The name for the new data group. Defaults to None.
            compression (FileAbstraction.Compression, optional): The compression method to use for the data. Defaults to FileAbstraction.Compression.DEFAULT.
            sparse (bool, optional): If False, creates non-sparse data (4D PSD array - new format). 
                                    If True or None, creates sparse data (2D flattened PSD array - old format for backward compatibility). 
                                    Defaults to None (uses old sparse format).
        Returns:
            Data: The newly created Data object.
        Raises:
            IndexError: If the specified index already exists in the dataset.
            ValueError: If any of the data provided is not valid or consistent
        """
        if PSD.ndim != 4:
            raise ValueError(
                "'PSD' must have 4 dimensions (z, y, x, spectrum)")
        if frequency.ndim != 1 and frequency.ndim != 4:
            raise ValueError(
                "'frequency' must have either 4 dimensions (z, y, x, spectrum) or 1 dimension (spectrum)")
        if len(px_size_um) != 3:
            raise ValueError("'px_size_um' must have 3 elements (z,y,x); unused dimensions can be set to nan")
        
        # Default to sparse=True (old format) for backward compatibility
        if sparse is None:
            sparse = True
        
        if sparse:
            # Sparse format (old): flatten PSD and create Cartesian_visualisation
            PSD_flat = np.reshape(PSD, (-1, PSD.shape[3]))
            if frequency.ndim == 4:
                freq_flat = np.reshape(frequency, (-1, frequency.shape[3]))
            else:
                freq_flat = frequency
            indices = np.arange(PSD_flat.shape[0])
            cartesian_vis = np.reshape(indices, PSD.shape[0:3])
            scanning = {'Cartesian_visualisation': cartesian_vis,
                       'Cartesian_visualisation_pixel': px_size_um, 
                       'Cartesian_visualisation_pixel_unit': 'um'}
            
            return self.create_data_group_raw(PSD_flat, freq_flat, scanning, 
                                             index=index, name=name, 
                                             compression=compression,
                                             sparse=True)
        else:
            # Non-sparse format (new): store PSD as 4D directly
            return self.create_data_group_raw(PSD, frequency, {}, 
                                             element_size=px_size_um, 
                                             element_size_unit='um',
                                             index=index, name=name, 
                                             compression=compression, 
                                             sparse=False)

    def create_data_group_raw(self, PSD: np.ndarray, frequency: np.ndarray, scanning: dict, 
                              element_size: tuple = None, element_size_unit: str = None,
                              timestamp: np.ndarray = None, index: int = None, name: str = None, 
                              compression: FileAbstraction.Compression = FileAbstraction.Compression(),
                              sparse: bool = None) -> 'Data':
        """
        Adds a new data entry to the file. Check the documentation for `brimfile.data.Data.add_data` for more details on the parameters.
        Parameters:
            PSD (np.ndarray): The Power Spectral Density (PSD) data to be added. The last dimension contains the spectra.
            frequency (np.ndarray): The frequency data corresponding to the PSD.
            scanning (dict): Metadata related to the scanning process. See Data.add_data for more details.
            element_size (tuple, optional): Pixel size for z, y, x dimensions. Used for non-sparse data.
            element_size_unit (str, optional): Unit for element_size. Defaults to None.
            timestamp (np.ndarray, optional): Timestamps in milliseconds for the data. Defaults to None.
            index (int, optional): The index for the new data group. If None, the next available index is used. Defaults to None.
            name (str, optional): The name for the new data group. Defaults to None.
            compression (FileAbstraction.Compression, optional): The compression method to use for the data. Defaults to FileAbstraction.Compression.DEFAULT.
            sparse (bool, optional): If True, sets Sparse attribute to true. If False, sets to false. If None, doesn't set the attribute (defaults to false). Defaults to None.
        Returns:
            Data: The newly created Data object.
        Raises:
            IndexError: If the specified index already exists in the dataset.
            ValueError: If any of the data provided is not valid or consistent
        """
        if index is not None:
            if Data._get_existing_group_name(self._file, index) is not None:
                raise IndexError(
                    f"Data {index} already exists in {self._file.filename}")
        else:
            data_groups = self.list_data_groups()
            indices = [dg['index'] for dg in data_groups]
            indices.sort()
            index = indices[-1] + 1 if indices else 0  # Next available index

        d = Data._create_new(self._file, index, name)
        d.add_data(PSD, frequency, scanning,
                   element_size=element_size,
                   element_size_unit=element_size_unit,
                   timestamp=timestamp, compression=compression, sparse=sparse)
        return d

    def list_data_groups(self, retrieve_custom_name=False) -> list:
        """
        List all data groups in the brim file.

        Returns:
            See documentation of brimfile.data.Data.list_data_groups
        """
        return Data.list_data_groups(self._file, retrieve_custom_name)

    def get_data(self, index: int = 0) -> 'Data':
        """
        Retrieve a Data object for the specified index.

        Args:
            index (int): The index of the data group to retrieve.

        Returns:
            Data: The Data object corresponding to the specified index.
        """
        group_name: str = Data._get_existing_group_name(self._file, index)
        if group_name is None:
            raise IndexError(f"Data {index} not found")
        data = Data(self._file, concatenate_paths(
            brim_obj_names.Brillouin_base_path, group_name))
        return data

    @property
    def filename(self) -> str:
        """
        Get the filename of the brim file.

        Returns:
            str: The filename of the brim file.
        """
        return self._file.filename
