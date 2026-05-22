# Prevent __pycache__ / .pyc creation — keeps Dropbox sync clean
import sys
sys.dont_write_bytecode = True

# Windows: inject the system certificate store into Python's SSL so that
# integration tests can reach HuggingFace without cert verification errors.
# Mirrors the same fix in whisperapp/__main__.py.
if sys.platform == "win32":
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        try:
            import certifi, os
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass
