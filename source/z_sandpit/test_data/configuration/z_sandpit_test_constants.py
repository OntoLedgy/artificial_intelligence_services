from pathlib import Path
from time import strftime, gmtime

Z_SANDPIT_TEST_DATA_FOLDER_PATH = Path(__file__).parent.parent.__str__()

COMPACT_TIMESTAMP_SUFFIX = strftime("_%Y%m%d_%H%M%S", gmtime())
