from npgru.preprocessor import *


def test_s3_download():
    preparer = ModelFilePreparer()
    preparer.prepare()
    assert preparer.every_compressed_file_exists(), "One of compressed file does not exist in model_dir"


def test_s3_decompress():
    decompressor = ModelFileDecompressor()
    decompressor.decompress()
    assert decompressor.every_decompressed_file_exists(), "One of compressed file is not decompressed in model_dir"
