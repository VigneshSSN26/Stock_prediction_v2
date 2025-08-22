import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  error: string | null;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Check if user is already logged in on app start
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        // For development: Comment out auto-login to always show login page
        // Uncomment the lines below to enable auto-login
        /*
        const token = localStorage.getItem('authToken');
        if (token) {
          // In a real app, you would validate the token with your backend
          // For now, we'll simulate a user session
          const userData = localStorage.getItem('userData');
          if (userData) {
            setUser(JSON.parse(userData));
          }
        }
        */
        
        // For development: Always start with no user logged in
        setUser(null);
      } catch (err) {
        console.error('Error checking auth status:', err);
        localStorage.removeItem('authToken');
        localStorage.removeItem('userData');
      } finally {
        setIsLoading(false);
      }
    };

    checkAuthStatus();
  }, []);

  const login = async (email: string, password: string) => {
    try {
      setIsLoading(true);
      setError(null);

      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      // In a real app, you would make an API call to your backend
      // For demo purposes, we'll simulate a successful login
      if (email === 'demo@example.com' && password === 'password') {
        const userData: User = {
          id: '1',
          name: 'Demo User',
          email: email,
          avatar: 'https://via.placeholder.com/40'
        };

        const token = 'demo-token-' + Date.now();
        
        localStorage.setItem('authToken', token);
        localStorage.setItem('userData', JSON.stringify(userData));
        
        setUser(userData);
      } else {
        throw new Error('Invalid email or password');
      }
    } catch (err: any) {
      setError(err.message || 'Login failed. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const signup = async (name: string, email: string, password: string) => {
    try {
      setIsLoading(true);
      setError(null);

      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1500));

      // In a real app, you would make an API call to your backend
      // For demo purposes, we'll simulate a successful signup
      const userData: User = {
        id: Date.now().toString(),
        name: name,
        email: email,
        avatar: 'https://via.placeholder.com/40'
      };

      const token = 'demo-token-' + Date.now();
      
      localStorage.setItem('authToken', token);
      localStorage.setItem('userData', JSON.stringify(userData));
      
      setUser(userData);
    } catch (err: any) {
      setError(err.message || 'Signup failed. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('userData');
    setUser(null);
    setError(null);
  };

  const clearError = () => {
    setError(null);
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    signup,
    logout,
    error,
    clearError
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
