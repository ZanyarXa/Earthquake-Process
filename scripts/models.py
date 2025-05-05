from sqlalchemy import Column, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Earthquake(Base):
    __tablename__ = 'earthquakes'
    id = Column(String, primary_key=True)
    time = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    depth = Column(Float)
    magnitude = Column(Float)
    magnitude_type = Column(String)
    type = Column(String)
    location_source = Column(String)
    source = Column(String)
    magnitude_source = Column(String)
    status = Column(String)

