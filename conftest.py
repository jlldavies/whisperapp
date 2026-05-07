# Prevent __pycache__ / .pyc creation — keeps Dropbox sync clean
import sys
sys.dont_write_bytecode = True
