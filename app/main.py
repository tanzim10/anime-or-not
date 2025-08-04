from fastapi import FastAPI

from app.routers.modelRouter import router

# Create an instance of the FastAPI application
app = FastAPI()

# Include routers
app.include_router(router)


def main() -> None:
    """
    Entry point for the application when run explicitly.
    """
    print("main.py running explicitly")


if __name__ == "__main__":
    main()

#Create an instance of the FastApi 