from npgru.decompressor.model_file import ModelFileDecompressor

decompressor = ModelFileDecompressor()


def test_s3_download():
    assert decompressor.every_compressed_file_exists(), "One of compressed file does not exist in model_dir"


def test_s3_decompress():
    decompressor.decompress()
    assert decompressor.every_decompressed_file_exists(), "One of compressed file is not decompressed in model_dir"
