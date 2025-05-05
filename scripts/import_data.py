from db_connection import create_db_session
from models import Earthquake
from preprocess import preprocess_data

def import_data_to_db():
    session = create_db_session()
    

    session.query(Earthquake).delete()
    

    df = preprocess_data()


    for _, row in df.iterrows():
        earthquake = Earthquake(**row.to_dict())
        session.add(earthquake)

    session.commit()
    session.close()
    print("Data import completed successfully!")

if __name__ == "__main__":
    import_data_to_db()

