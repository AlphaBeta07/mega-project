/**
 * StudySnap AI — Centralized Frontend API Configuration
 * Supports environment-based configuration for production and local environments.
 */

export const API_GATEWAY_URL =
  import.meta.env.VITE_API_GATEWAY_URL || 'http://localhost:5000';

export const AI_BACKEND_URL =
  import.meta.env.VITE_AI_BACKEND_URL || 'http://localhost:8000';

/**
 * Retrieve saved auth token from localStorage
 */
export const getAuthToken = (): string | null => {
  try {
    return localStorage.getItem('studysnap_auth_token');
  } catch {
    return null;
  }
};

/**
 * Set auth token in localStorage
 */
export const setAuthToken = (token: string | null): void => {
  try {
    if (token) {
      localStorage.setItem('studysnap_auth_token', token);
    } else {
      localStorage.removeItem('studysnap_auth_token');
    }
  } catch (err) {
    console.warn('Failed to save token to localStorage:', err);
  }
};

/**
 * Remove auth token on logout
 */
export const removeAuthToken = (): void => {
  try {
    localStorage.removeItem('studysnap_auth_token');
    localStorage.removeItem('studysnap_user');
  } catch (err) {
    console.warn('Failed to clear auth from localStorage:', err);
  }
};

/**
 * Authenticated fetch helper for Main Backend API Gateway
 */
export const apiFetch = async (endpoint: string, options: RequestInit = {}): Promise<Response> => {
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});

  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const url = endpoint.startsWith('http') ? endpoint : `${API_GATEWAY_URL}${endpoint}`;
  return fetch(url, { ...options, headers });
};

/**
 * Authenticated fetch helper for Python FastAPI AI Backend (port 8000)
 */
export const aiFetch = async (endpoint: string, options: RequestInit = {}): Promise<Response> => {
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});

  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const url = endpoint.startsWith('http') ? endpoint : `${AI_BACKEND_URL}${endpoint}`;
  return fetch(url, { ...options, headers });
};

