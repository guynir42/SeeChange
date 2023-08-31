"""
Here we put all the dictionaries and convertion functions for getting/setting enums and bitflags.
"""

# This is the master format dictionary, that contains all file types for
# all data models. Each model will get a subset of this dictionary.
file_format_dict = {
    1: 'fits',
    2: 'hdf5',
    3: 'csv',
    4: 'json',
    5: 'yaml',
    6: 'xml',
    7: 'pickle',
    8: 'parquet',
    9: 'npy',
    10: 'npz',
    11: 'avro',
    12: 'netcdf',
    13: 'jpg',
    14: 'png',
    15: 'pdf',
}

allowed_image_formats = ['fits', 'hdf5']
image_format_dict = {k: v for k, v in file_format_dict.items() if v in allowed_image_formats}
image_format_inverse = {v: k for k, v in image_format_dict.items()}


def image_format_converter(value):
    if isinstance(value, str):
        if value not in image_format_inverse:
            raise ValueError(f'Image type must be one of {image_format_inverse.keys()}, not {value}')
        return image_format_inverse[value]
    elif isinstance(value, (int, float)):
        if value not in image_format_dict:
            raise ValueError(f'Image type integer key must be one of {image_format_dict.keys()}, not {value}')
        return image_format_dict[value]
    elif value is None:
        return None
    else:
        raise ValueError(f'Image type must be integer/float key or string value, not {type(value)}')


allowed_cutout_formats = ['fits', 'hdf5', 'jpg', 'png']
cutouts_format_dict = {k: v for k, v in file_format_dict.items() if v in allowed_cutout_formats}
cutouts_format_inverse = {v: k for k, v in cutouts_format_dict.items()}


def cutouts_format_converter(value):
    if isinstance(value, str):
        if value not in cutouts_format_inverse:
            raise ValueError(f'Cutout type must be one of {cutouts_format_inverse.keys()}, not {value}')
        return cutouts_format_inverse[value]
    elif isinstance(value, (int, float)):
        if value not in cutouts_format_dict:
            raise ValueError(f'Cutout type integer key must be one of {cutouts_format_dict.keys()}, not {value}')
        return cutouts_format_dict[value]
    elif value is None:
        return None
    else:
        raise ValueError(f'Cutout type must be integer/float key or string value, not {type(value)}')


allowed_source_list_formats = ['npy', 'csv', 'hdf5', 'parquet', 'fits']
source_list_format_dict = {k: v for k, v in file_format_dict.items() if v in allowed_source_list_formats}
source_list_format_inverse = {v: k for k, v in source_list_format_dict.items()}


def source_list_format_converter(value):
    if isinstance(value, str):
        if value not in source_list_format_inverse:
            raise ValueError(f'Source list type must be one of {source_list_format_inverse.keys()}, not {value}')
        return source_list_format_inverse[value]
    elif isinstance(value, (int, float)):
        if value not in source_list_format_dict:
            raise ValueError(f'Source list type integer key must be one of {source_list_format_dict.keys()}, not {value}')
        return source_list_format_dict[value]
    elif value is None:
        return None
    else:
        raise ValueError(f'Source list type must be integer/float key or string value, not {type(value)}')


image_type_dict = {
    1: 'Sci',
    2: 'ComSci',
    3: 'Diff',
    4: 'ComDiff',
    5: 'Bias',
    6: 'ComBias',
    7: 'Dark',
    8: 'ComDark',
    9: 'DomeFlat',
    10: 'ComDomeFlat',
    11: 'SkyFlat',
    12: 'ComSkyFlat',
    13: 'TwiFlat',
    14: 'ComTwiFlat',
}
image_type_inverse = {v: k for k, v in image_type_dict.items()}


def image_type_converter(value):
    if isinstance(value, str):
        if value not in image_type_inverse:
            raise ValueError(f'Image type must be one of {image_type_inverse.keys()}, not {value}')
        return image_type_inverse[value]
    elif isinstance(value, (int, float)):
        if value not in image_type_dict:
            raise ValueError(f'Image type integer key must be one of {image_type_dict.keys()}, not {value}')
        return image_type_dict[value]
    elif value is None:
        return None
    else:
        raise ValueError(f'Image type must be integer/float key or string value, not {type(value)}')



