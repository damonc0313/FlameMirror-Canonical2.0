#!/usr/bin/env python3
"""Production-grade data validator with comprehensive error handling."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of data validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)


class DataValidator:
    """Validates data against schema with detailed error reporting."""
    
    def __init__(self, schema: Dict[str, type], strict: bool = True):
        self.schema = schema
        self.strict = strict
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data record against schema."""
        result = ValidationResult(valid=True)
        
        # Check required fields
        for field, expected_type in self.schema.items():
            if field not in data:
                result.add_error(f"Missing required field: {field}")
                continue
            
            # Type checking
            value = data[field]
            if not isinstance(value, expected_type):
                result.add_error(
                    f"Type mismatch: {field} should be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
        
        # Check for extra fields in strict mode
        if self.strict:
            for field in data:
                if field not in self.schema:
                    result.add_warning(f"Unexpected field: {field}")
        
        return result
    
    def validate_batch(self, records: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate multiple records."""
        return [self.validate(record) for record in records]


if __name__ == '__main__':
    # Example usage
    schema = {
        'name': str,
        'age': int,
        'email': str
    }
    
    validator = DataValidator(schema)
    
    # Valid data
    result = validator.validate({'name': 'Alice', 'age': 30, 'email': 'alice@example.com'})
    print(f"Valid: {result.valid}")
    
    # Invalid data
    result = validator.validate({'name': 'Bob', 'age': 'thirty'})
    print(f"Valid: {result.valid}, Errors: {result.errors}")
