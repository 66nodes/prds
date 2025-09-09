"""
Audit System Performance Tests

Tests audit system performance under various load conditions,
including high-volume logging, concurrent operations, and stress testing.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch
import uuid

import pytest
import structlog

from services.comprehensive_audit_service import (
    ComprehensiveAuditService,
    AuditEventType,
    AuditSeverity,
    ComplianceStandard
)
from core.audit_middleware import AuditMiddleware


logger = structlog.get_logger(__name__)


@pytest.fixture
async def performance_audit_service():
    """Create audit service configured for performance testing."""
    service = ComprehensiveAuditService(
        batch_size=50,  # Smaller batch for testing
        buffer_size=1000,
        flush_interval=0.1,  # Fast flushing for tests
        retention_days=1  # Short retention for tests
    )
    
    # Mock Redis for performance testing
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock()
    mock_redis.get = AsyncMock()
    mock_redis.keys = AsyncMock(return_value=[])
    mock_redis.delete = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    
    with patch('services.comprehensive_audit_service.get_redis_client', return_value=mock_redis):
        await service.initialize()
        yield service, mock_redis
        await service.cleanup()


class TestAuditServicePerformance:
    """Test audit service performance characteristics."""
    
    async def test_high_volume_logging_performance(self, performance_audit_service):
        """Test performance with high volume of audit events."""
        service, mock_redis = performance_audit_service
        
        # Test parameters
        num_events = 1000
        max_duration = 10.0  # seconds
        
        # Generate events concurrently
        start_time = time.time()
        
        tasks = []
        for i in range(num_events):
            task = service.log_audit_event(
                event_type=AuditEventType.DATA_READ,
                user_id=f"perf-user-{i % 10}",
                resource_type="prd",
                resource_id=f"prd-{i}",
                outcome="success",
                metadata={"performance_test": True, "event_number": i}
            )
            tasks.append(task)
        
        # Execute all tasks
        event_ids = await asyncio.gather(*tasks)
        end_time = time.time()
        
        duration = end_time - start_time
        events_per_second = num_events / duration
        
        # Performance assertions
        assert duration < max_duration, f"High volume logging took {duration:.2f}s, expected < {max_duration}s"
        assert events_per_second > 100, f"Only {events_per_second:.2f} events/sec, expected > 100"
        assert len(event_ids) == num_events
        assert all(event_id is not None for event_id in event_ids)
        
        logger.info(
            "High volume logging performance",
            events=num_events,
            duration_seconds=duration,
            events_per_second=events_per_second
        )
    
    async def test_concurrent_logging_performance(self, performance_audit_service):
        """Test concurrent logging from multiple coroutines."""
        service, mock_redis = performance_audit_service
        
        async def worker(worker_id: int, events_per_worker: int) -> List[str]:
            """Worker coroutine that logs events."""
            event_ids = []
            for i in range(events_per_worker):
                event_id = await service.log_audit_event(
                    event_type=AuditEventType.USER_LOGIN,
                    user_id=f"worker-{worker_id}-user-{i}",
                    outcome="success",
                    metadata={"worker_id": worker_id, "event_index": i}
                )
                event_ids.append(event_id)
            return event_ids
        
        # Test parameters
        num_workers = 10
        events_per_worker = 100
        total_events = num_workers * events_per_worker
        max_duration = 15.0
        
        # Start concurrent workers
        start_time = time.time()
        
        worker_tasks = [
            worker(worker_id, events_per_worker)
            for worker_id in range(num_workers)
        ]
        
        worker_results = await asyncio.gather(*worker_tasks)
        end_time = time.time()
        
        duration = end_time - start_time
        events_per_second = total_events / duration
        
        # Verify results
        all_event_ids = []
        for worker_events in worker_results:
            all_event_ids.extend(worker_events)
        
        # Performance assertions
        assert duration < max_duration, f"Concurrent logging took {duration:.2f}s, expected < {max_duration}s"
        assert events_per_second > 50, f"Only {events_per_second:.2f} events/sec, expected > 50"
        assert len(all_event_ids) == total_events
        assert len(set(all_event_ids)) == total_events, "Event IDs should be unique"
        
        logger.info(
            "Concurrent logging performance",
            workers=num_workers,
            events_per_worker=events_per_worker,
            total_events=total_events,
            duration_seconds=duration,
            events_per_second=events_per_second
        )
    
    async def test_batch_processing_performance(self, performance_audit_service):
        """Test batch processing performance."""
        service, mock_redis = performance_audit_service
        
        # Generate many events quickly to fill buffer
        num_events = 500
        
        # Log events rapidly
        start_time = time.time()
        
        event_tasks = [
            service.log_audit_event(
                event_type=AuditEventType.DATA_UPDATED,
                user_id=f"batch-user-{i % 5}",
                resource_id=f"resource-{i}",
                outcome="success"
            )
            for i in range(num_events)
        ]
        
        await asyncio.gather(*event_tasks)
        
        # Wait for buffer to flush
        await asyncio.sleep(1.0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify batch processing occurred
        assert mock_redis.setex.call_count > 0
        assert mock_redis.setex.call_count < num_events  # Should be batched, not individual
        
        # Should process all events efficiently
        assert duration < 5.0, f"Batch processing took {duration:.2f}s, expected < 5s"
        
        logger.info(
            "Batch processing performance",
            events=num_events,
            redis_calls=mock_redis.setex.call_count,
            duration_seconds=duration,
            batch_efficiency=num_events / mock_redis.setex.call_count
        )
    
    async def test_search_performance(self, performance_audit_service):
        """Test search performance with large datasets."""
        service, mock_redis = performance_audit_service
        
        # Mock large search results
        num_mock_events = 10000
        mock_events = [
            {
                "event_id": f"search-perf-{i}",
                "timestamp": (datetime.now(timezone.utc) - timedelta(seconds=i)).isoformat(),
                "event_type": AuditEventType.DATA_READ.value,
                "user_id": f"search-user-{i % 100}",
                "outcome": "success"
            }
            for i in range(num_mock_events)
        ]
        
        # Mock Redis responses
        mock_redis.keys.return_value = [f"audit_event:search-perf-{i}" for i in range(num_mock_events)]
        mock_redis.get.side_effect = [json.dumps(event) for event in mock_events]
        
        # Test search performance
        start_time = time.time()
        
        events, total_count = await service.search_audit_events(
            event_types=[AuditEventType.DATA_READ],
            limit=1000,
            offset=0
        )
        
        end_time = time.time()
        search_duration = end_time - start_time
        
        # Performance assertions
        assert search_duration < 2.0, f"Search took {search_duration:.2f}s, expected < 2s"
        assert len(events) <= 1000  # Respects limit
        assert total_count > 0
        
        logger.info(
            "Search performance",
            mock_events=num_mock_events,
            returned_events=len(events),
            search_duration_seconds=search_duration
        )
    
    async def test_compliance_report_generation_performance(self, performance_audit_service):
        """Test compliance report generation performance."""
        service, mock_redis = performance_audit_service
        
        # Mock compliance-relevant events
        num_events = 5000
        compliance_events = []
        
        for i in range(num_events):
            event_type = [
                AuditEventType.USER_LOGIN,
                AuditEventType.USER_LOGOUT,
                AuditEventType.DATA_READ,
                AuditEventType.DATA_UPDATED,
                AuditEventType.ADMIN_ACTION
            ][i % 5]
            
            compliance_events.append({
                "event_id": f"compliance-{i}",
                "timestamp": (datetime.now(timezone.utc) - timedelta(seconds=i)).isoformat(),
                "event_type": event_type.value,
                "severity": AuditSeverity.MEDIUM.value,
                "compliance_tags": ["soc2", "security"],
                "outcome": "success"
            })
        
        # Mock Redis for compliance report
        mock_redis.keys.return_value = [f"audit_event:compliance-{i}" for i in range(num_events)]
        mock_redis.get.side_effect = [json.dumps(event) for event in compliance_events]
        
        # Test report generation performance
        start_time = time.time()
        
        report = await service.generate_compliance_report(
            standard=ComplianceStandard.SOC2_TYPE2,
            period_start=datetime.now(timezone.utc) - timedelta(days=30),
            period_end=datetime.now(timezone.utc)
        )
        
        end_time = time.time()
        report_duration = end_time - start_time
        
        # Performance assertions
        assert report_duration < 5.0, f"Report generation took {report_duration:.2f}s, expected < 5s"
        assert report is not None
        assert report.total_events > 0
        assert report.compliance_score >= 0.0
        
        logger.info(
            "Compliance report performance",
            events_analyzed=num_events,
            report_duration_seconds=report_duration,
            compliance_score=report.compliance_score
        )
    
    async def test_retention_policy_performance(self, performance_audit_service):
        """Test retention policy application performance."""
        service, mock_redis = performance_audit_service
        
        # Mock events with mixed ages
        num_events = 2000
        old_events = 800
        recent_events = 1200
        
        # Mock old events (beyond retention)
        old_keys = [f"audit_event:old-{i}" for i in range(old_events)]
        recent_keys = [f"audit_event:recent-{i}" for i in range(recent_events)]
        all_keys = old_keys + recent_keys
        
        mock_redis.scan_iter.return_value = all_keys
        
        # Mock timestamp responses
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        recent_timestamp = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        
        def mock_get(key):
            if "old-" in key:
                return json.dumps({"timestamp": old_timestamp})
            else:
                return json.dumps({"timestamp": recent_timestamp})
        
        mock_redis.get.side_effect = mock_get
        
        # Test retention policy performance
        start_time = time.time()
        
        summary = await service.apply_retention_policy()
        
        end_time = time.time()
        retention_duration = end_time - start_time
        
        # Performance assertions
        assert retention_duration < 3.0, f"Retention policy took {retention_duration:.2f}s, expected < 3s"
        assert summary is not None
        assert "events_deleted" in summary
        assert "events_retained" in summary
        
        # Verify deletions occurred
        assert mock_redis.delete.call_count > 0
        
        logger.info(
            "Retention policy performance",
            total_events=num_events,
            retention_duration_seconds=retention_duration,
            delete_operations=mock_redis.delete.call_count
        )


class TestAuditMiddlewarePerformance:
    """Test audit middleware performance impact."""
    
    async def test_middleware_request_processing_overhead(self):
        """Test middleware overhead on request processing."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        # Create app without middleware
        app_no_middleware = FastAPI()
        
        @app_no_middleware.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        # Create app with audit middleware
        app_with_middleware = FastAPI()
        
        @app_with_middleware.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        # Mock audit service
        mock_audit_service = AsyncMock()
        mock_audit_service.log_audit_event = AsyncMock(return_value="test-event")
        
        # Add middleware
        middleware = AuditMiddleware(app_with_middleware)
        middleware.audit_service = mock_audit_service
        
        # Create clients
        client_no_middleware = TestClient(app_no_middleware)
        client_with_middleware = TestClient(middleware)
        
        # Warm up
        for _ in range(10):
            client_no_middleware.get("/test")
            client_with_middleware.get("/test")
        
        # Benchmark without middleware
        num_requests = 100
        
        start_time = time.time()
        for _ in range(num_requests):
            response = client_no_middleware.get("/test")
            assert response.status_code == 200
        no_middleware_duration = time.time() - start_time
        
        # Benchmark with middleware
        start_time = time.time()
        for _ in range(num_requests):
            response = client_with_middleware.get("/test")
            assert response.status_code == 200
        with_middleware_duration = time.time() - start_time
        
        # Calculate overhead
        overhead = with_middleware_duration - no_middleware_duration
        overhead_percentage = (overhead / no_middleware_duration) * 100
        
        # Performance assertions
        assert overhead_percentage < 50, f"Middleware adds {overhead_percentage:.1f}% overhead, expected < 50%"
        
        logger.info(
            "Middleware performance overhead",
            requests=num_requests,
            no_middleware_duration=no_middleware_duration,
            with_middleware_duration=with_middleware_duration,
            overhead_seconds=overhead,
            overhead_percentage=overhead_percentage
        )
    
    async def test_middleware_concurrent_request_handling(self):
        """Test middleware performance with concurrent requests."""
        from fastapi import FastAPI
        from httpx import AsyncClient
        
        app = FastAPI()
        
        @app.get("/concurrent-test/{item_id}")
        async def concurrent_endpoint(item_id: str):
            # Simulate some processing time
            await asyncio.sleep(0.01)
            return {"item_id": item_id, "processed": True}
        
        # Mock audit service
        mock_audit_service = AsyncMock()
        mock_audit_service.log_audit_event = AsyncMock(return_value="concurrent-event")
        
        # Add middleware
        middleware = AuditMiddleware(app)
        middleware.audit_service = mock_audit_service
        
        # Test concurrent requests
        num_concurrent_requests = 50
        
        async with AsyncClient(app=middleware, base_url="http://test") as client:
            start_time = time.time()
            
            tasks = [
                client.get(f"/concurrent-test/{i}")
                for i in range(num_concurrent_requests)
            ]
            
            responses = await asyncio.gather(*tasks)
            
            end_time = time.time()
            concurrent_duration = end_time - start_time
        
        # Verify all requests succeeded
        assert all(response.status_code == 200 for response in responses)
        assert len(responses) == num_concurrent_requests
        
        # Performance assertions
        assert concurrent_duration < 5.0, f"Concurrent requests took {concurrent_duration:.2f}s, expected < 5s"
        
        # Should have logged audit events for all requests
        assert mock_audit_service.log_audit_event.call_count >= num_concurrent_requests
        
        logger.info(
            "Middleware concurrent performance",
            concurrent_requests=num_concurrent_requests,
            duration_seconds=concurrent_duration,
            requests_per_second=num_concurrent_requests / concurrent_duration
        )


class TestAuditSystemStressTest:
    """Stress test the complete audit system."""
    
    async def test_sustained_high_load(self, performance_audit_service):
        """Test system under sustained high load."""
        service, mock_redis = performance_audit_service
        
        # Stress test parameters
        duration_seconds = 10
        target_events_per_second = 100
        total_events = duration_seconds * target_events_per_second
        
        # Track timing and success
        start_time = time.time()
        successful_events = 0
        failed_events = 0
        
        # Generate sustained load
        event_tasks = []
        for i in range(total_events):
            # Vary event types and data
            event_type = [
                AuditEventType.USER_LOGIN,
                AuditEventType.DATA_READ,
                AuditEventType.DATA_CREATED,
                AuditEventType.DATA_UPDATED,
                AuditEventType.ADMIN_ACTION
            ][i % 5]
            
            task = service.log_audit_event(
                event_type=event_type,
                user_id=f"stress-user-{i % 20}",
                resource_id=f"resource-{i}",
                outcome="success",
                metadata={"stress_test": True, "event_number": i}
            )
            event_tasks.append(task)
        
        # Execute all events
        try:
            event_ids = await asyncio.gather(*event_tasks, return_exceptions=True)
            
            # Count successes and failures
            for event_id in event_ids:
                if isinstance(event_id, Exception):
                    failed_events += 1
                elif event_id is not None:
                    successful_events += 1
                else:
                    failed_events += 1
                    
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            failed_events = total_events
        
        end_time = time.time()
        actual_duration = end_time - start_time
        actual_events_per_second = successful_events / actual_duration
        success_rate = (successful_events / total_events) * 100
        
        # Stress test assertions
        assert success_rate > 95, f"Only {success_rate:.1f}% success rate, expected > 95%"
        assert actual_events_per_second > 50, f"Only {actual_events_per_second:.1f} events/sec, expected > 50"
        
        logger.info(
            "Stress test results",
            total_events=total_events,
            successful_events=successful_events,
            failed_events=failed_events,
            success_rate=success_rate,
            duration_seconds=actual_duration,
            actual_events_per_second=actual_events_per_second,
            target_events_per_second=target_events_per_second
        )
    
    async def test_memory_usage_under_load(self, performance_audit_service):
        """Test memory usage doesn't grow excessively under load."""
        service, mock_redis = performance_audit_service
        
        # Monitor initial buffer size
        initial_buffer_size = len(service.event_buffer)
        
        # Generate load in batches to test memory management
        batch_size = 100
        num_batches = 10
        
        max_buffer_size_seen = initial_buffer_size
        
        for batch in range(num_batches):
            # Generate batch of events
            batch_tasks = [
                service.log_audit_event(
                    event_type=AuditEventType.DATA_READ,
                    user_id=f"memory-user-{i}",
                    resource_id=f"batch-{batch}-item-{i}",
                    outcome="success"
                )
                for i in range(batch_size)
            ]
            
            await asyncio.gather(*batch_tasks)
            
            # Monitor buffer size
            current_buffer_size = len(service.event_buffer)
            max_buffer_size_seen = max(max_buffer_size_seen, current_buffer_size)
            
            # Allow time for buffer to flush
            await asyncio.sleep(0.2)
        
        # Final buffer size should be managed
        final_buffer_size = len(service.event_buffer)
        
        # Memory management assertions
        assert max_buffer_size_seen < service.buffer_size, "Buffer size should not exceed configured limit"
        assert final_buffer_size <= initial_buffer_size + 50, "Buffer should not grow excessively"
        
        logger.info(
            "Memory usage test",
            initial_buffer_size=initial_buffer_size,
            max_buffer_size_seen=max_buffer_size_seen,
            final_buffer_size=final_buffer_size,
            configured_buffer_limit=service.buffer_size
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])