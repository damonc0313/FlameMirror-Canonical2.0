#!/usr/bin/env python3
"""
AUTH & AUTHORIZATION SERVICE - Iteration 2.1
=============================================
JWT-based authentication with RBAC
"""

import jwt
import hashlib
from typing import Dict, Set, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    roles: Set[str]
    permissions: Set[str]

class AuthService:
    """Complete auth system with JWT, RBAC, password hashing"""
    
    def __init__(self, secret_key: str):
        self.secret = secret_key
        self.users: Dict[str, User] = {}
        self.role_permissions: Dict[str, Set[str]] = {}
        
    def register(self, username: str, password: str, roles: Set[str]) -> User:
        """Register new user"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        user = User(
            user_id=hashlib.md5(username.encode()).hexdigest(),
            username=username,
            password_hash=password_hash,
            roles=roles,
            permissions=self._get_role_permissions(roles)
        )
        self.users[user.user_id] = user
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate and return JWT token"""
        user = next((u for u in self.users.values() if u.username == username), None)
        if not user:
            return None
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash != user.password_hash:
            return None
        
        # Generate JWT
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': list(user.roles),
            'exp': (datetime.now(timezone.utc) + timedelta(hours=24)).timestamp()
        }
        
        token = jwt.encode(payload, self.secret, algorithm='HS256')
        return token
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def authorize(self, token: str, required_permission: str) -> bool:
        """Check if user has permission"""
        payload = self.verify_token(token)
        if not payload:
            return False
        
        user = self.users.get(payload['user_id'])
        return required_permission in user.permissions if user else False
    
    def _get_role_permissions(self, roles: Set[str]) -> Set[str]:
        """Get all permissions for roles"""
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        return permissions

# 100 lines - Focused auth service
