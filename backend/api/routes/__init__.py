from .crop_yield import router as crop_yield_router
from .disease import router as disease
from .resource import router as resource
from .soil import router as soil_router
from .weather import router as weather

__all__ = ['disease', 'resource', 'soil_router', 'weather', 'crop_yield_router']
