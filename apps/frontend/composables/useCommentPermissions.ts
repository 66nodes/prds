import { computed } from 'vue'
import { useAuth } from './useAuth'
import type { Comment, User } from '~/types'

interface PermissionConfig {
  // Global permissions
  canCreateComments?: boolean
  canEditAnyComment?: boolean
  canDeleteAnyComment?: boolean
  canModerateComments?: boolean
  canViewPrivateComments?: boolean
  canAssignComments?: boolean
  canManageReactions?: boolean
  
  // Document-level permissions
  canViewDocument?: boolean
  canCommentOnDocument?: boolean
  canEditDocument?: boolean
  canManageDocument?: boolean
  
  // Role-based permissions
  roles?: string[]
  permissions?: string[]
}

interface CommentPermissions {
  // Comment actions
  canView: boolean
  canEdit: boolean
  canDelete: boolean
  canReply: boolean
  canReact: boolean
  canResolve: boolean
  canAssign: boolean
  canChangeStatus: boolean
  
  // UI permissions
  canShowEditButton: boolean
  canShowDeleteButton: boolean
  canShowResolveButton: boolean
  canShowAssignButton: boolean
  canShowStatusSelect: boolean
  
  // Metadata
  isAuthor: boolean
  isMentioned: boolean
  isAssigned: boolean
  hasModeratorRights: boolean
}

export const useCommentPermissions = () => {
  const { user } = useAuth()

  // Default permission configuration
  const defaultPermissions: PermissionConfig = {
    canCreateComments: true,
    canEditAnyComment: false,
    canDeleteAnyComment: false,
    canModerateComments: false,
    canViewPrivateComments: false,
    canAssignComments: false,
    canManageReactions: true,
    canViewDocument: true,
    canCommentOnDocument: true,
    canEditDocument: false,
    canManageDocument: false,
    roles: [],
    permissions: []
  }

  // Get user permissions from user object or API
  const getUserPermissions = (): PermissionConfig => {
    if (!user.value) {
      return {
        ...defaultPermissions,
        canCreateComments: false,
        canCommentOnDocument: false
      }
    }

    // Merge default permissions with user-specific permissions
    const userPermissions: PermissionConfig = {
      ...defaultPermissions,
      ...(user.value.permissions || {}),
      roles: user.value.roles || [],
      permissions: user.value.user_permissions || []
    }

    // Apply role-based permissions
    if (userPermissions.roles?.includes('admin') || userPermissions.roles?.includes('superuser')) {
      return {
        ...userPermissions,
        canEditAnyComment: true,
        canDeleteAnyComment: true,
        canModerateComments: true,
        canViewPrivateComments: true,
        canAssignComments: true,
        canEditDocument: true,
        canManageDocument: true
      }
    }

    if (userPermissions.roles?.includes('moderator')) {
      return {
        ...userPermissions,
        canModerateComments: true,
        canViewPrivateComments: true,
        canAssignComments: true,
        canEditAnyComment: true
      }
    }

    if (userPermissions.roles?.includes('editor')) {
      return {
        ...userPermissions,
        canEditDocument: true,
        canAssignComments: true
      }
    }

    return userPermissions
  }

  // Computed user permissions
  const permissions = computed(getUserPermissions)

  // Check if user has a specific permission
  const hasPermission = (permission: string): boolean => {
    if (!user.value) return false
    
    const userPerms = permissions.value
    
    // Check direct permissions
    if (userPerms.permissions?.includes(permission)) {
      return true
    }
    
    // Check role-based permissions
    if (userPerms.roles?.includes('admin') || userPerms.roles?.includes('superuser')) {
      return true
    }
    
    // Check specific permission mappings
    const permissionMap: Record<string, boolean> = {
      'comment:create': userPerms.canCreateComments || false,
      'comment:edit_any': userPerms.canEditAnyComment || false,
      'comment:delete_any': userPerms.canDeleteAnyComment || false,
      'comment:moderate': userPerms.canModerateComments || false,
      'comment:view_private': userPerms.canViewPrivateComments || false,
      'comment:assign': userPerms.canAssignComments || false,
      'comment:manage_reactions': userPerms.canManageReactions || false,
      'document:view': userPerms.canViewDocument || false,
      'document:comment': userPerms.canCommentOnDocument || false,
      'document:edit': userPerms.canEditDocument || false,
      'document:manage': userPerms.canManageDocument || false
    }
    
    return permissionMap[permission] || false
  }

  // Check if user has any of the specified roles
  const hasRole = (roles: string | string[]): boolean => {
    if (!user.value || !permissions.value.roles) return false
    
    const checkRoles = Array.isArray(roles) ? roles : [roles]
    return checkRoles.some(role => permissions.value.roles?.includes(role))
  }

  // Get permissions for a specific comment
  const getCommentPermissions = (comment: Comment, documentPermissions?: PermissionConfig): CommentPermissions => {
    if (!user.value) {
      return {
        canView: false,
        canEdit: false,
        canDelete: false,
        canReply: false,
        canReact: false,
        canResolve: false,
        canAssign: false,
        canChangeStatus: false,
        canShowEditButton: false,
        canShowDeleteButton: false,
        canShowResolveButton: false,
        canShowAssignButton: false,
        canShowStatusSelect: false,
        isAuthor: false,
        isMentioned: false,
        isAssigned: false,
        hasModeratorRights: false
      }
    }

    const userPerms = permissions.value
    const docPerms = documentPermissions || userPerms
    
    const isAuthor = comment.author_id === user.value.id
    const isMentioned = comment.mentions?.includes(user.value.id) || false
    const isAssigned = comment.assignees?.includes(user.value.id) || false
    const hasModeratorRights = hasRole(['admin', 'superuser', 'moderator'])
    
    // Basic view permission
    const canView = (() => {
      if (!docPerms.canViewDocument) return false
      if (comment.is_private) {
        return isAuthor || isMentioned || isAssigned || userPerms.canViewPrivateComments
      }
      return true
    })()

    // Edit permissions
    const canEdit = canView && (isAuthor || userPerms.canEditAnyComment || hasModeratorRights)
    
    // Delete permissions
    const canDelete = canView && (isAuthor || userPerms.canDeleteAnyComment || hasModeratorRights)
    
    // Reply permissions
    const canReply = canView && docPerms.canCommentOnDocument && userPerms.canCreateComments
    
    // Reaction permissions
    const canReact = canView && userPerms.canManageReactions
    
    // Resolution permissions (author, assignees, or moderators)
    const canResolve = canView && (isAuthor || isAssigned || hasModeratorRights || userPerms.canModerateComments)
    
    // Assignment permissions
    const canAssign = canView && (userPerms.canAssignComments || hasModeratorRights)
    
    // Status change permissions
    const canChangeStatus = canView && (isAuthor || isAssigned || hasModeratorRights || userPerms.canModerateComments)

    return {
      canView,
      canEdit,
      canDelete,
      canReply,
      canReact,
      canResolve,
      canAssign,
      canChangeStatus,
      
      // UI permissions (more restrictive for cleaner UI)
      canShowEditButton: canEdit && (isAuthor || hasModeratorRights),
      canShowDeleteButton: canDelete && (isAuthor || hasModeratorRights),
      canShowResolveButton: canResolve && (comment.status === 'open' || comment.status === 'in_progress'),
      canShowAssignButton: canAssign,
      canShowStatusSelect: canChangeStatus && (isAuthor || isAssigned || hasModeratorRights),
      
      // Metadata
      isAuthor,
      isMentioned,
      isAssigned,
      hasModeratorRights
    }
  }

  // Get document-level permissions for comments
  const getDocumentCommentPermissions = (documentId: string): PermissionConfig => {
    // This would typically fetch document-specific permissions from an API
    // For now, return the user's global permissions
    return permissions.value
  }

  // Check if user can perform bulk operations
  const canPerformBulkOperations = (comments: Comment[]): boolean => {
    if (!user.value || comments.length === 0) return false
    
    // User must be able to moderate or be the author of all comments
    if (hasRole(['admin', 'superuser', 'moderator'])) {
      return true
    }
    
    // Check if user is the author of all comments
    return comments.every(comment => comment.author_id === user.value!.id)
  }

  // Check if user can view comment analytics
  const canViewAnalytics = (documentId?: string): boolean => {
    return hasRole(['admin', 'superuser', 'moderator', 'editor']) || 
           hasPermission('document:manage') ||
           hasPermission('comment:moderate')
  }

  // Check if user can export comments
  const canExportComments = (documentId?: string): boolean => {
    return hasRole(['admin', 'superuser', 'moderator', 'editor']) || 
           hasPermission('document:view')
  }

  // Get permission error message
  const getPermissionErrorMessage = (action: string, comment?: Comment): string => {
    const messages: Record<string, string> = {
      'view': 'You do not have permission to view this comment.',
      'edit': 'You do not have permission to edit this comment.',
      'delete': 'You do not have permission to delete this comment.',
      'reply': 'You do not have permission to reply to comments.',
      'react': 'You do not have permission to react to comments.',
      'resolve': 'You do not have permission to resolve this comment.',
      'assign': 'You do not have permission to assign comments.',
      'changeStatus': 'You do not have permission to change comment status.',
      'create': 'You do not have permission to create comments.',
      'moderate': 'You do not have permission to moderate comments.',
      'viewPrivate': 'You do not have permission to view private comments.',
      'bulkOperation': 'You do not have permission to perform bulk operations.',
      'analytics': 'You do not have permission to view analytics.',
      'export': 'You do not have permission to export comments.'
    }
    
    return messages[action] || 'You do not have permission to perform this action.'
  }

  // Permission validation functions
  const validateCommentAction = (action: string, comment: Comment): { allowed: boolean; message?: string } => {
    const commentPerms = getCommentPermissions(comment)
    const actionMap: Record<string, boolean> = {
      'view': commentPerms.canView,
      'edit': commentPerms.canEdit,
      'delete': commentPerms.canDelete,
      'reply': commentPerms.canReply,
      'react': commentPerms.canReact,
      'resolve': commentPerms.canResolve,
      'assign': commentPerms.canAssign,
      'changeStatus': commentPerms.canChangeStatus
    }
    
    const allowed = actionMap[action] || false
    
    return {
      allowed,
      message: allowed ? undefined : getPermissionErrorMessage(action, comment)
    }
  }

  return {
    // Core permissions
    permissions: readonly(permissions),
    
    // Permission checking
    hasPermission,
    hasRole,
    
    // Comment-specific permissions
    getCommentPermissions,
    getDocumentCommentPermissions,
    validateCommentAction,
    
    // Bulk operations
    canPerformBulkOperations,
    
    // Analytics and export
    canViewAnalytics,
    canExportComments,
    
    // Error messages
    getPermissionErrorMessage,
    
    // User state
    isAuthenticated: computed(() => !!user.value),
    currentUser: readonly(user)
  }
}