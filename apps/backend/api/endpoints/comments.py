"""
API endpoints for Comment and Annotation System.

Provides REST API for collaborative feedback on planning documents including
inline comments, annotations, threaded replies, and real-time updates.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from ...core.database import get_db_session, get_redis
from ...services.auth_service import get_current_user, User
from ...services.websocket_manager import WebSocketManager, get_websocket_manager
from ...models.comments import (
    Comment,
    CommentCreate,
    CommentUpdate,
    CommentType,
    CommentStatus,
    CommentPriority,
    DocumentType,
    CommentListResponse,
    CommentSummary,
    CommentThread,
    CommentNotification,
    CommentAnalytics,
    CommentSearchRequest,
    CommentBatchOperation,
    CommentExportRequest,
    CommentReaction
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comments", tags=["comments"])
security = HTTPBearer()


# Mock storage for demonstration - in production, use proper database
comment_storage: Dict[str, Comment] = {}
thread_storage: Dict[str, List[str]] = {}  # thread_id -> [comment_ids]


async def get_comment_service():
    """Get comment service instance."""
    # In production, this would return actual service instance
    return None


def generate_comment_id() -> str:
    """Generate unique comment ID."""
    import uuid
    return str(uuid.uuid4())


def generate_thread_id() -> str:
    """Generate unique thread ID."""
    import uuid
    return str(uuid.uuid4())


def calculate_comment_depth(parent_id: Optional[str]) -> int:
    """Calculate comment nesting depth."""
    if not parent_id or parent_id not in comment_storage:
        return 0
    
    parent = comment_storage[parent_id]
    return min(parent.depth + 1, 10)  # Max depth of 10


def build_comment_thread(thread_id: str) -> CommentThread:
    """Build comment thread with nested structure."""
    thread_comments = [
        comment for comment in comment_storage.values() 
        if comment.thread_id == thread_id
    ]
    
    if not thread_comments:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Find root comment
    root_comment = next((c for c in thread_comments if c.parent_id is None), None)
    if not root_comment:
        raise HTTPException(status_code=404, detail="Root comment not found")
    
    def build_replies(parent_id: str) -> List[CommentThread]:
        replies = []
        child_comments = [c for c in thread_comments if c.parent_id == parent_id]
        child_comments.sort(key=lambda x: x.created_at)
        
        for child in child_comments:
            child_thread = CommentThread(
                root_comment=child,
                replies=build_replies(child.id),
                total_replies=len([c for c in thread_comments if c.parent_id == child.id]),
                participants=list(set(c.author_id for c in thread_comments)),
                last_activity=max(c.last_activity for c in thread_comments)
            )
            replies.append(child_thread)
        
        return replies
    
    return CommentThread(
        root_comment=root_comment,
        replies=build_replies(root_comment.id),
        total_replies=len(thread_comments) - 1,
        participants=list(set(c.author_id for c in thread_comments)),
        last_activity=max(c.last_activity for c in thread_comments)
    )


async def send_comment_notification(
    comment: Comment,
    action: str,
    websocket_manager: WebSocketManager,
    background_tasks: BackgroundTasks
):
    """Send real-time notification for comment actions."""
    try:
        # Send WebSocket update
        message_data = {
            "action": action,
            "comment": {
                "id": comment.id,
                "content": comment.content[:100] + "..." if len(comment.content) > 100 else comment.content,
                "author_name": comment.author_name,
                "document_id": comment.document_id,
                "created_at": comment.created_at.isoformat(),
                "comment_type": comment.comment_type.value,
                "status": comment.status.value
            }
        }
        
        # Notify document collaborators
        await websocket_manager.broadcast_to_document(
            document_id=comment.document_id,
            message_type="comment_update",
            data=message_data
        )
        
        # Notify mentioned users
        for user_id in comment.mentions:
            await websocket_manager.send_to_user(
                user_id=user_id,
                message_type="user_mentioned",
                data=message_data
            )
        
        # Notify assigned users
        for user_id in comment.assignees:
            await websocket_manager.send_to_user(
                user_id=user_id,
                message_type="comment_assigned",
                data=message_data
            )
        
        logger.info(f"Sent comment notification for {action}", extra={
            "comment_id": comment.id,
            "document_id": comment.document_id,
            "action": action
        })
        
    except Exception as e:
        logger.error(f"Failed to send comment notification: {str(e)}")


@router.post("/", response_model=Comment)
async def create_comment(
    comment_data: CommentCreate,
    current_user: User = Depends(get_current_user),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create a new comment or annotation."""
    try:
        comment_id = generate_comment_id()
        
        # Determine thread ID
        if comment_data.parent_id:
            # Reply to existing comment
            if comment_data.parent_id not in comment_storage:
                raise HTTPException(status_code=404, detail="Parent comment not found")
            
            parent_comment = comment_storage[comment_data.parent_id]
            thread_id = parent_comment.thread_id
            depth = calculate_comment_depth(comment_data.parent_id)
            
            # Update parent reply count
            parent_comment.reply_count += 1
            parent_comment.last_activity = datetime.utcnow()
        else:
            # New thread
            thread_id = generate_thread_id()
            depth = 0
        
        # Create comment
        comment = Comment(
            id=comment_id,
            thread_id=thread_id,
            depth=depth,
            document_id=comment_data.document_id,
            document_type=comment_data.document_type,
            author_id=current_user.id,
            author_name=current_user.full_name,
            author_avatar=None,  # Could be retrieved from user service
            parent_id=comment_data.parent_id,
            content=comment_data.content,
            comment_type=comment_data.comment_type,
            priority=comment_data.priority,
            is_private=comment_data.is_private,
            tags=comment_data.tags,
            selection_range=comment_data.selection_range,
            position=comment_data.position,
            mentions=comment_data.mentions,
            assignees=comment_data.assignees,
            status=CommentStatus.OPEN,
            reply_count=0
        )
        
        # Store comment
        comment_storage[comment_id] = comment
        
        # Track thread
        if thread_id not in thread_storage:
            thread_storage[thread_id] = []
        thread_storage[thread_id].append(comment_id)
        
        # Send notifications
        await send_comment_notification(
            comment=comment,
            action="created",
            websocket_manager=websocket_manager,
            background_tasks=background_tasks
        )
        
        logger.info(f"Created comment {comment_id}", extra={
            "comment_id": comment_id,
            "document_id": comment_data.document_id,
            "author_id": current_user.id,
            "comment_type": comment_data.comment_type.value
        })
        
        return comment
        
    except Exception as e:
        logger.error(f"Failed to create comment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create comment: {str(e)}"
        )


@router.get("/{comment_id}", response_model=Comment)
async def get_comment(
    comment_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific comment by ID."""
    if comment_id not in comment_storage:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    comment = comment_storage[comment_id]
    
    # Check if user can view private comments
    if comment.is_private and comment.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied to private comment")
    
    return comment


@router.put("/{comment_id}", response_model=Comment)
async def update_comment(
    comment_id: str,
    comment_update: CommentUpdate,
    current_user: User = Depends(get_current_user),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Update an existing comment."""
    if comment_id not in comment_storage:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    comment = comment_storage[comment_id]
    
    # Check permissions (author or admin can edit)
    if comment.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to edit this comment")
    
    # Update fields
    update_data = comment_update.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        if hasattr(comment, field):
            setattr(comment, field, value)
    
    # Update timestamps
    comment.updated_at = datetime.utcnow()
    comment.last_activity = datetime.utcnow()
    
    # Handle status changes
    if comment_update.status == CommentStatus.RESOLVED and comment.status != CommentStatus.RESOLVED:
        comment.resolved_by = current_user.id
        comment.resolved_at = datetime.utcnow()
        if comment_update.resolution_note:
            comment.resolution_note = comment_update.resolution_note
    
    # Send notifications
    await send_comment_notification(
        comment=comment,
        action="updated",
        websocket_manager=websocket_manager,
        background_tasks=background_tasks
    )
    
    logger.info(f"Updated comment {comment_id}", extra={
        "comment_id": comment_id,
        "updated_by": current_user.id,
        "changes": list(update_data.keys())
    })
    
    return comment


@router.delete("/{comment_id}")
async def delete_comment(
    comment_id: str,
    current_user: User = Depends(get_current_user),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Delete a comment (soft delete or hard delete based on permissions)."""
    if comment_id not in comment_storage:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    comment = comment_storage[comment_id]
    
    # Check permissions
    if comment.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this comment")
    
    # Check if comment has replies
    has_replies = any(c.parent_id == comment_id for c in comment_storage.values())
    
    if has_replies:
        # Soft delete - mark as deleted but keep for thread integrity
        comment.content = "[This comment has been deleted]"
        comment.status = CommentStatus.CLOSED
        comment.updated_at = datetime.utcnow()
        action = "soft_deleted"
    else:
        # Hard delete - remove completely
        del comment_storage[comment_id]
        
        # Remove from thread storage
        for thread_comments in thread_storage.values():
            if comment_id in thread_comments:
                thread_comments.remove(comment_id)
                break
        
        # Update parent reply count
        if comment.parent_id and comment.parent_id in comment_storage:
            parent = comment_storage[comment.parent_id]
            parent.reply_count = max(0, parent.reply_count - 1)
        
        action = "deleted"
    
    # Send notifications
    await send_comment_notification(
        comment=comment,
        action=action,
        websocket_manager=websocket_manager,
        background_tasks=background_tasks
    )
    
    logger.info(f"Deleted comment {comment_id}", extra={
        "comment_id": comment_id,
        "deleted_by": current_user.id,
        "action": action
    })
    
    return {"message": f"Comment {action} successfully"}


@router.get("/document/{document_id}", response_model=CommentListResponse)
async def list_document_comments(
    document_id: str,
    current_user: User = Depends(get_current_user),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    status: Optional[CommentStatus] = Query(None, description="Filter by status"),
    comment_type: Optional[CommentType] = Query(None, description="Filter by type"),
    include_private: bool = Query(False, description="Include private comments"),
    threaded: bool = Query(True, description="Return threaded structure")
):
    """List all comments for a document."""
    # Filter comments for document
    document_comments = [
        comment for comment in comment_storage.values()
        if comment.document_id == document_id
    ]
    
    # Apply filters
    if status:
        document_comments = [c for c in document_comments if c.status == status]
    
    if comment_type:
        document_comments = [c for c in document_comments if c.comment_type == comment_type]
    
    # Handle private comments
    if not include_private:
        document_comments = [
            c for c in document_comments 
            if not c.is_private or c.author_id == current_user.id
        ]
    
    # Sort by creation time
    document_comments.sort(key=lambda x: x.created_at, reverse=True)
    
    # Pagination for flat list
    total_count = len(document_comments)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_comments = document_comments[start_idx:end_idx]
    
    # Create summaries
    comment_summaries = [
        CommentSummary(
            id=comment.id,
            content=comment.content[:200] + "..." if len(comment.content) > 200 else comment.content,
            comment_type=comment.comment_type,
            priority=comment.priority,
            status=comment.status,
            author_name=comment.author_name,
            reply_count=comment.reply_count,
            created_at=comment.created_at,
            last_activity=comment.last_activity
        )
        for comment in page_comments
    ]
    
    # Build threads if requested
    threads = []
    if threaded:
        # Get unique thread IDs from current page
        thread_ids = list(set(c.thread_id for c in page_comments if c.parent_id is None))
        
        for thread_id in thread_ids:
            try:
                thread = build_comment_thread(thread_id)
                threads.append(thread)
            except HTTPException:
                continue  # Skip broken threads
    
    # Calculate counts
    open_count = len([c for c in document_comments if c.status == CommentStatus.OPEN])
    resolved_count = len([c for c in document_comments if c.status == CommentStatus.RESOLVED])
    
    return CommentListResponse(
        comments=comment_summaries,
        threads=threads,
        total_count=total_count,
        open_count=open_count,
        resolved_count=resolved_count,
        page=page,
        page_size=page_size,
        has_more=end_idx < total_count
    )


@router.get("/thread/{thread_id}", response_model=CommentThread)
async def get_comment_thread(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    include_private: bool = Query(False, description="Include private comments")
):
    """Get a complete comment thread with all replies."""
    try:
        thread = build_comment_thread(thread_id)
        
        # Filter private comments if needed
        if not include_private:
            def filter_private(thread_node: CommentThread) -> CommentThread:
                # Filter the root comment
                if thread_node.root_comment.is_private and thread_node.root_comment.author_id != current_user.id:
                    raise HTTPException(status_code=403, detail="Access denied to private comment")
                
                # Filter replies recursively
                filtered_replies = []
                for reply in thread_node.replies:
                    try:
                        filtered_reply = filter_private(reply)
                        filtered_replies.append(filtered_reply)
                    except HTTPException:
                        continue  # Skip private replies
                
                thread_node.replies = filtered_replies
                return thread_node
            
            thread = filter_private(thread)
        
        return thread
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get comment thread: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get comment thread: {str(e)}"
        )


@router.post("/{comment_id}/reactions")
async def add_reaction(
    comment_id: str,
    reaction_type: str,
    current_user: User = Depends(get_current_user),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """Add or update a reaction to a comment."""
    if comment_id not in comment_storage:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    comment = comment_storage[comment_id]
    
    # Check if user already reacted
    existing_reaction = next(
        (r for r in comment.reactions if r.user_id == current_user.id),
        None
    )
    
    if existing_reaction:
        # Update existing reaction
        existing_reaction.reaction_type = reaction_type
    else:
        # Add new reaction
        reaction = CommentReaction(
            id=generate_comment_id(),
            user_id=current_user.id,
            user_name=current_user.full_name,
            reaction_type=reaction_type
        )
        comment.reactions.append(reaction)
    
    comment.last_activity = datetime.utcnow()
    
    # Send real-time update
    await websocket_manager.broadcast_to_document(
        document_id=comment.document_id,
        message_type="comment_reaction",
        data={
            "comment_id": comment_id,
            "reaction_type": reaction_type,
            "user_name": current_user.full_name,
            "total_reactions": len(comment.reactions)
        }
    )
    
    return {"message": "Reaction added successfully"}


@router.delete("/{comment_id}/reactions")
async def remove_reaction(
    comment_id: str,
    current_user: User = Depends(get_current_user),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """Remove a user's reaction from a comment."""
    if comment_id not in comment_storage:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    comment = comment_storage[comment_id]
    
    # Find and remove user's reaction
    original_count = len(comment.reactions)
    comment.reactions = [r for r in comment.reactions if r.user_id != current_user.id]
    
    if len(comment.reactions) == original_count:
        raise HTTPException(status_code=404, detail="No reaction found to remove")
    
    comment.last_activity = datetime.utcnow()
    
    # Send real-time update
    await websocket_manager.broadcast_to_document(
        document_id=comment.document_id,
        message_type="comment_reaction_removed",
        data={
            "comment_id": comment_id,
            "user_name": current_user.full_name,
            "total_reactions": len(comment.reactions)
        }
    )
    
    return {"message": "Reaction removed successfully"}


@router.post("/search", response_model=CommentListResponse)
async def search_comments(
    search_request: CommentSearchRequest,
    current_user: User = Depends(get_current_user),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size")
):
    """Search comments with advanced filtering."""
    # Start with all comments
    filtered_comments = list(comment_storage.values())
    
    # Apply filters
    if search_request.query:
        query_lower = search_request.query.lower()
        filtered_comments = [
            c for c in filtered_comments
            if query_lower in c.content.lower() or 
               query_lower in c.author_name.lower() or
               any(query_lower in tag.lower() for tag in c.tags)
        ]
    
    if search_request.document_id:
        filtered_comments = [c for c in filtered_comments if c.document_id == search_request.document_id]
    
    if search_request.document_type:
        filtered_comments = [c for c in filtered_comments if c.document_type == search_request.document_type]
    
    if search_request.author_id:
        filtered_comments = [c for c in filtered_comments if c.author_id == search_request.author_id]
    
    if search_request.comment_type:
        filtered_comments = [c for c in filtered_comments if c.comment_type == search_request.comment_type]
    
    if search_request.status:
        filtered_comments = [c for c in filtered_comments if c.status == search_request.status]
    
    if search_request.priority:
        filtered_comments = [c for c in filtered_comments if c.priority == search_request.priority]
    
    if search_request.date_from:
        filtered_comments = [c for c in filtered_comments if c.created_at >= search_request.date_from]
    
    if search_request.date_to:
        filtered_comments = [c for c in filtered_comments if c.created_at <= search_request.date_to]
    
    if search_request.tags:
        filtered_comments = [
            c for c in filtered_comments
            if any(tag in c.tags for tag in search_request.tags)
        ]
    
    if search_request.mentions_user:
        filtered_comments = [c for c in filtered_comments if search_request.mentions_user in c.mentions]
    
    if search_request.assigned_to:
        filtered_comments = [c for c in filtered_comments if search_request.assigned_to in c.assignees]
    
    if search_request.has_selection is not None:
        if search_request.has_selection:
            filtered_comments = [c for c in filtered_comments if c.selection_range is not None]
        else:
            filtered_comments = [c for c in filtered_comments if c.selection_range is None]
    
    if not search_request.include_resolved:
        filtered_comments = [c for c in filtered_comments if c.status != CommentStatus.RESOLVED]
    
    # Filter private comments
    filtered_comments = [
        c for c in filtered_comments
        if not c.is_private or c.author_id == current_user.id
    ]
    
    # Sort results
    if search_request.sort_by == "created_at":
        filtered_comments.sort(key=lambda x: x.created_at, reverse=(search_request.sort_order == "desc"))
    elif search_request.sort_by == "updated_at":
        filtered_comments.sort(key=lambda x: x.updated_at, reverse=(search_request.sort_order == "desc"))
    elif search_request.sort_by == "priority":
        priority_order = {CommentPriority.CRITICAL: 4, CommentPriority.HIGH: 3, CommentPriority.MEDIUM: 2, CommentPriority.LOW: 1}
        filtered_comments.sort(key=lambda x: priority_order.get(x.priority, 0), reverse=(search_request.sort_order == "desc"))
    
    # Pagination
    total_count = len(filtered_comments)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_comments = filtered_comments[start_idx:end_idx]
    
    # Create summaries
    comment_summaries = [
        CommentSummary(
            id=comment.id,
            content=comment.content[:200] + "..." if len(comment.content) > 200 else comment.content,
            comment_type=comment.comment_type,
            priority=comment.priority,
            status=comment.status,
            author_name=comment.author_name,
            reply_count=comment.reply_count,
            created_at=comment.created_at,
            last_activity=comment.last_activity
        )
        for comment in page_comments
    ]
    
    # Calculate counts
    open_count = len([c for c in filtered_comments if c.status == CommentStatus.OPEN])
    resolved_count = len([c for c in filtered_comments if c.status == CommentStatus.RESOLVED])
    
    return CommentListResponse(
        comments=comment_summaries,
        threads=[],  # Not building threads for search results
        total_count=total_count,
        open_count=open_count,
        resolved_count=resolved_count,
        page=page,
        page_size=page_size,
        has_more=end_idx < total_count
    )


@router.get("/analytics/{document_id}", response_model=CommentAnalytics)
async def get_comment_analytics(
    document_id: str,
    current_user: User = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365, description="Number of days for analytics")
):
    """Get analytics data for document comments."""
    document_comments = [
        comment for comment in comment_storage.values()
        if comment.document_id == document_id
    ]
    
    if not document_comments:
        return CommentAnalytics(
            document_id=document_id,
            total_comments=0,
            open_comments=0,
            resolved_comments=0,
            average_resolution_time_hours=0,
            top_commenters=[],
            comment_types_distribution={},
            activity_timeline=[]
        )
    
    # Calculate metrics
    open_comments = [c for c in document_comments if c.status == CommentStatus.OPEN]
    resolved_comments = [c for c in document_comments if c.status == CommentStatus.RESOLVED]
    
    # Calculate average resolution time
    resolution_times = []
    for comment in resolved_comments:
        if comment.resolved_at:
            resolution_time = (comment.resolved_at - comment.created_at).total_seconds() / 3600
            resolution_times.append(resolution_time)
    
    avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
    
    # Top commenters
    commenter_counts = {}
    for comment in document_comments:
        commenter_counts[comment.author_name] = commenter_counts.get(comment.author_name, 0) + 1
    
    top_commenters = [
        {"name": name, "count": count}
        for name, count in sorted(commenter_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    
    # Comment types distribution
    type_distribution = {}
    for comment in document_comments:
        type_name = comment.comment_type.value
        type_distribution[type_name] = type_distribution.get(type_name, 0) + 1
    
    # Activity timeline (simplified)
    from_date = datetime.utcnow() - timedelta(days=days)
    recent_comments = [c for c in document_comments if c.created_at >= from_date]
    
    # Group by day
    daily_activity = {}
    for comment in recent_comments:
        day_key = comment.created_at.strftime("%Y-%m-%d")
        daily_activity[day_key] = daily_activity.get(day_key, 0) + 1
    
    activity_timeline = [
        {"date": date, "count": count}
        for date, count in sorted(daily_activity.items())
    ]
    
    return CommentAnalytics(
        document_id=document_id,
        total_comments=len(document_comments),
        open_comments=len(open_comments),
        resolved_comments=len(resolved_comments),
        average_resolution_time_hours=avg_resolution_time,
        top_commenters=top_commenters,
        comment_types_distribution=type_distribution,
        activity_timeline=activity_timeline
    )


@router.post("/batch", response_model=Dict[str, Any])
async def batch_update_comments(
    batch_operation: CommentBatchOperation,
    current_user: User = Depends(get_current_user),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """Perform batch operations on multiple comments."""
    results = {"success": [], "failed": [], "errors": []}
    
    for comment_id in batch_operation.comment_ids:
        try:
            if comment_id not in comment_storage:
                results["failed"].append(comment_id)
                results["errors"].append(f"Comment {comment_id} not found")
                continue
            
            comment = comment_storage[comment_id]
            
            # Check permissions
            if comment.author_id != current_user.id:
                results["failed"].append(comment_id)
                results["errors"].append(f"Not authorized to modify comment {comment_id}")
                continue
            
            # Perform operation
            if batch_operation.operation == "resolve":
                comment.status = CommentStatus.RESOLVED
                comment.resolved_by = current_user.id
                comment.resolved_at = datetime.utcnow()
                if "resolution_note" in batch_operation.parameters:
                    comment.resolution_note = batch_operation.parameters["resolution_note"]
            
            elif batch_operation.operation == "close":
                comment.status = CommentStatus.CLOSED
                
            elif batch_operation.operation == "assign":
                if "assignees" in batch_operation.parameters:
                    comment.assignees = batch_operation.parameters["assignees"]
                    
            elif batch_operation.operation == "tag":
                if "tags" in batch_operation.parameters:
                    comment.tags = list(set(comment.tags + batch_operation.parameters["tags"]))
            
            else:
                results["failed"].append(comment_id)
                results["errors"].append(f"Unknown operation: {batch_operation.operation}")
                continue
            
            comment.updated_at = datetime.utcnow()
            comment.last_activity = datetime.utcnow()
            
            results["success"].append(comment_id)
            
            # Send notification
            await send_comment_notification(
                comment=comment,
                action=f"batch_{batch_operation.operation}",
                websocket_manager=websocket_manager,
                background_tasks=BackgroundTasks()
            )
            
        except Exception as e:
            results["failed"].append(comment_id)
            results["errors"].append(f"Error processing {comment_id}: {str(e)}")
    
    return results


@router.get("/health", response_model=Dict[str, Any])
async def comment_system_health():
    """Health check for comment system."""
    return {
        "status": "healthy",
        "service": "comment-system",
        "timestamp": datetime.utcnow().isoformat(),
        "statistics": {
            "total_comments": len(comment_storage),
            "total_threads": len(thread_storage),
            "active_threads": len([t for t in thread_storage.values() if t])
        },
        "version": "1.0.0"
    }