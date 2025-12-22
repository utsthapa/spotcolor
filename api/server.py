"""Local development server for SpotColor API.

This module provides a simple entry point for running the API locally
with hot-reload enabled for development.

Usage:
    python -m api.server

Or using the installed script:
    spotcolor-api
"""

import uvicorn


def main():
    """Run the development server with hot-reload enabled.

    Starts uvicorn on http://0.0.0.0:8000 with automatic reload
    when source files change in the api/ or screenprint/ directories.
    """
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["api", "screenprint"]
    )


if __name__ == "__main__":
    main()
