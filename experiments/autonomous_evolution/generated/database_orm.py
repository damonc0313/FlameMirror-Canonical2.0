#!/usr/bin/env python3
"""
DATABASE ORM - Iteration 2.2
=============================
Lightweight ORM with migrations, queries, transactions
"""

from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass, fields
import sqlite3
import json

class Model:
    """Base model class"""
    def to_dict(self) -> Dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})

class Database:
    """Database connection and query builder"""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL query"""
        return self.conn.execute(query, params)
    
    def commit(self):
        """Commit transaction"""
        self.conn.commit()
    
    def rollback(self):
        """Rollback transaction"""
        self.conn.rollback()

class QueryBuilder:
    """SQL query builder"""
    
    def __init__(self, table: str):
        self.table = table
        self.wheres: List[str] = []
        self.params: List[Any] = []
        self.limit_val: Optional[int] = None
        self.order_val: Optional[str] = None
        
    def where(self, column: str, value: Any):
        """Add WHERE clause"""
        self.wheres.append(f"{column} = ?")
        self.params.append(value)
        return self
    
    def limit(self, n: int):
        """Add LIMIT clause"""
        self.limit_val = n
        return self
    
    def order_by(self, column: str, direction: str = 'ASC'):
        """Add ORDER BY clause"""
        self.order_val = f"{column} {direction}"
        return self
    
    def build_select(self) -> tuple:
        """Build SELECT query"""
        query = f"SELECT * FROM {self.table}"
        
        if self.wheres:
            query += " WHERE " + " AND ".join(self.wheres)
        if self.order_val:
            query += f" ORDER BY {self.order_val}"
        if self.limit_val:
            query += f" LIMIT {self.limit_val}"
        
        return query, tuple(self.params)

class Repository:
    """Generic repository pattern"""
    
    def __init__(self, db: Database, model_class: Type[Model], table: str):
        self.db = db
        self.model_class = model_class
        self.table = table
        
    def find_by_id(self, id: Any) -> Optional[Model]:
        """Find by ID"""
        query, params = QueryBuilder(self.table).where('id', id).build_select()
        cursor = self.db.execute(query, params)
        row = cursor.fetchone()
        return self.model_class.from_dict(dict(row)) if row else None
    
    def find_all(self, limit: int = 100) -> List[Model]:
        """Find all records"""
        query, params = QueryBuilder(self.table).limit(limit).build_select()
        cursor = self.db.execute(query, params)
        return [self.model_class.from_dict(dict(row)) for row in cursor.fetchall()]
    
    def save(self, model: Model):
        """Save model"""
        data = model.to_dict()
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT OR REPLACE INTO {self.table} ({columns}) VALUES ({placeholders})"
        self.db.execute(query, tuple(data.values()))
        self.db.commit()
    
    def delete(self, id: Any):
        """Delete by ID"""
        self.db.execute(f"DELETE FROM {self.table} WHERE id = ?", (id,))
        self.db.commit()

# 150 lines - Focused ORM
