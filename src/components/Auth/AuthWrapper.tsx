import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import Login from './Login';
import Signup from './Signup';
import './Auth.css';

const AuthWrapper: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);
  const { login, signup, isLoading, error, clearError } = useAuth();

  const handleLogin = async (email: string, password: string) => {
    try {
      await login(email, password);
    } catch (err) {
      // Error is handled by the auth context
    }
  };

  const handleSignup = async (name: string, email: string, password: string, confirmPassword: string) => {
    try {
      if (password !== confirmPassword) {
        throw new Error('Passwords do not match');
      }
      await signup(name, email, password);
    } catch (err: any) {
      // Error is handled by the auth context
    }
  };

  const switchToSignup = () => {
    setIsLogin(false);
    clearError();
  };

  const switchToLogin = () => {
    setIsLogin(true);
    clearError();
  };

  return (
    <>
      {isLogin ? (
        <Login
          onLogin={handleLogin}
          onSwitchToSignup={switchToSignup}
          isLoading={isLoading}
          error={error}
        />
      ) : (
        <Signup
          onSignup={handleSignup}
          onSwitchToLogin={switchToLogin}
          isLoading={isLoading}
          error={error}
        />
      )}
    </>
  );
};

export default AuthWrapper;
