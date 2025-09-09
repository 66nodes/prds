#!/usr/bin/env python3
"""
Test Data Factory - Deterministic Test Data Generation

Generates compliance-friendly test data that meets security policies
and quality gate requirements. No hardcoded placeholders allowed.
"""

import uuid
from faker import Faker
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC

class TestDataFactory:
    """Factory for generating compliant test data."""

    def __init__(self, seed: int = 42):
        """Initialize factory with deterministic seeding."""
        self.seed = seed
        self.fake = Faker()
        Faker.seed(seed)  # Deterministic seeding

    def generate_user(self, role: str = "user") -> Dict[str, Any]:
        """Generate a complete user object."""
        user_id = str(uuid.uuid4())

        return {
            "id": user_id,
            "email": f"{role.lower()}{self.fake.random_number(digits=4)}@{self.fake.domain_name()}",
            "full_name": self.fake.name(),
            "first_name": self.fake.first_name(),
            "last_name": self.fake.last_name(),
            "is_active": True,
            "is_superuser": role == "admin",
            "role": role,
            "department": self.fake.company(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat()
        }

    def generate_company(self) -> Dict[str, Any]:
        """Generate a company for testing."""
        return {
            "id": str(uuid.uuid4()),
            "name": self.fake.company(),
            "industry": self.fake.bs().capitalize(),
            "size": self.fake.random_int(min=10, max=10000),
            "website": self.fake.url(),
            "location": self.fake.city() + ", " + self.fake.state_abbr()
        }

    def generate_project(self, company_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a project for testing."""
        if not company_id:
            company_id = str(uuid.uuid4())

        return {
            "id": str(uuid.uuid4()),
            "name": self.fake.catch_phrase(),
            "description": self.fake.text(max_nb_chars=200),
            "company_id": company_id,
            "status": self.fake.random_element(["planning", "active", "completed", "on_hold"]),
            "budget": self.fake.random_int(min=10000, max=5000000),
            "start_date": self.fake.date_between(start_date="-1y", end_date="+1y").isoformat(),
            "end_date": self.fake.date_between(start_date="+1y", end_date="+2y").isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat()
        }

    def generate_prd(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a PRD for testing."""
        if not project_id:
            project_id = str(uuid.uuid4())

        return {
            "id": str(uuid.uuid4()),
            "title": self.fake.catch_phrase(),
            "project_id": project_id,
            "description": self.fake.paragraph(nb_sentences=5),
            "requirements": [self.fake.sentence(nb_words=10) for _ in range(5)],
            "acceptance_criteria": [self.fake.sentence(nb_words=8) for _ in range(3)],
            "stakeholders": [self.fake.name() for _ in range(3)],
            "status": self.fake.random_element(["draft", "review", "approved", "implemented"]),
            "priority": self.fake.random_element(["low", "medium", "high", "critical"]),
            "estimated_effort_hours": self.fake.random_int(min=40, max=320),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat()
        }

    def generate_batch_users(self, count: int, role: str = "user") -> List[Dict[str, Any]]:
        """Generate multiple users."""
        return [self.generate_user(role) for _ in range(count)]

    def generate_batch_projects(self, count: int) -> List[Dict[str, Any]]:
        """Generate multiple projects."""
        return [self.generate_project() for _ in range(count)]

    def generate_batch_prds(self, count: int) -> List[Dict[str, Any]]:
        """Generate multiple PRDs."""
        return [self.generate_prd() for _ in range(count)]


# Global factory instance for consistent data generation
test_data_factory = TestDataFactory()

if __name__ == "__main__":
    # Example usage
    factory = TestDataFactory()
    user = factory.generate_user("admin")
    project = factory.generate_project()
    print(f"Generated user: {user['email']}")
    print(f"Generated project: {project['name']}")