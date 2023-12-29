from typing import Optional

from sqlmodel import (
    Field, 
    SQLModel, 
    Session, 
    create_engine, 
    select , 
    func)
from dataclasses import dataclass
from tqdm import tqdm
import datetime

class Text(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, description = "The id of the text")
    text: str = Field(..., description = "text to analyze")
    is_parsed:bool = Field(default=False, description = "is the text parsed or analyzed")
    parse_result: Optional[str] = Field(default=None, description = "parsed result json string")
    key:Optional[str] = Field(default=None, description = "unique identifier of the text(from business logics)")
    created_at: Optional[int] = Field(default_factory = func.now, description = "created time")
    updated_at: Optional[int] = Field(default_factory = func.now, description = "updated time")
    
@dataclass
class TextDB:
    
    engine_url:str
    
    @property
    def engine(self):
        return create_engine(self.engine_url)
    
    def create_database(self, tables = None, checkfirst:bool = True):
        SQLModel.metadata.create_all(self.engine,tables = tables, checkfirst=checkfirst)
        print("Database and tables created.")
        
    def pd_add_dataframe(self, df, text_col_name:str = 'text', key_col_name:str = 'key',if_exists:str = 'append'):
        df = df[[text_col_name, key_col_name]]
        df = df.rename(columns = {text_col_name:'text', key_col_name:'key'})
        df['is_parsed'] = False
        df['parse_result'] = None
        dt = datetime.datetime.now().replace(microsecond=0)
        df['created_at'] = dt
        df['updated_at'] = dt
        df.to_sql('text', self.engine, if_exists=if_exists, index=False)
        print("successfully add all records to database")
            
    def add(self, data):
        with Session(self.engine) as session:
            session.add(data)
            session.commit()
            data = session.refresh(data)
        return data
    
    def delete_by_id(self, id):
        with Session(self.engine) as session:
            statement = select(Text).where(Text.id == id)
            data = session.exec(statement).one()
            session.delete(data)
            session.commit()
        return True
    
    def delete_by_key(self, key):
        with Session(self.engine) as session:
            statement = select(Text).where(Text.key == key)
            data = session.exec(statement).one()
            session.delete(data)
            session.commit()
        return True
    
    def get_num_unparsed(self):
        with Session(self.engine) as session:
            statement = select(Text).where(Text.is_parsed == False)
            data = session.exec(statement).all()
        return len(data)
    
    def update_by_id(self, id:int, parse_result:str):
        # if not find id in database
        with Session(self.engine) as session:
            if not session.exec(select(Text).where(Text.id == id)).one():
                return None
            statement = select(Text).where(Text.id == id)
            data = session.exec(statement).one()
            data.is_parsed = True
            data.parse_result = parse_result
            data.updated_at = datetime.datetime.now().replace(microsecond=0)
            session.add(data)
            session.commit()
            session.refresh(data)
        return data
    
    def update_by_key(self, key:str, parse_result:str):
        # if not find id in database
        with Session(self.engine) as session:
            if not session.exec(select(Text).where(Text.key == key)).one():
                return None
            statement = select(Text).where(Text.key == key)
            data = session.exec(statement).one()
            data.is_parsed = True
            data.parse_result = parse_result
            data.updated_at = datetime.datetime.now().replace(microsecond=0)
            session.add(data)
            session.commit()
            session.refresh(data)
        return data
    
    def add_dataframe(self, df, text_col_name:str = 'text', key_col_name:str = 'key'):
        for _, row in tqdm(df.iterrows(),total = len(df), ncols = 80):
            self.add(Text(text=row[text_col_name], key=row[key_col_name]))
        
        print("insert all rows from dataframe")
    
    def get_limit_offset(self, offset:int = 0, limit:int = 1):
        with Session(self.engine) as session:
            statement = select(Text).where(Text.is_parsed == False).offset(offset).limit(limit)
            data = session.exec(statement).all()
        return data
    
    def get_unparsed_limit_offset(self, offset:int = 0, limit:int = 1):
        with Session(self.engine) as session:
            statement = select(Text).where(Text.is_parsed == False).offset(offset).limit(limit)
            data = session.exec(statement).all()
        return data
    
    def delete_all(self):
        with Session(self.engine) as session:
            statement = select(Text)
            data = session.exec(statement).all()
            for d in data:
                session.delete(d)
            session.commit()
        return True
    
    def get_all(self):
        with Session(self.engine) as session:
            statement = select(Text)
            data = session.exec(statement).all()
        return data
    
    def get_by_id(self, id:int):
        with Session(self.engine) as session:
            statement = select(Text).where(Text.id == id)
            data = session.exec(statement).one_or_none()
        return data
    
    def get_by_key(self, key:str):
        with Session(self.engine) as session:
            statement = select(Text).where(Text.key == key)
            data = session.exec(statement).one_or_none()
        return data
