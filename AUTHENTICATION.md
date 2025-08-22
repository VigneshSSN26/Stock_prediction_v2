# Authentication System

This document describes the authentication system implemented in the StockAI Predictor frontend.

## Features

### Login Page
- Professional login form with email and password fields
- Password visibility toggle
- Remember me checkbox
- Social login options (Google, GitHub)
- Form validation and error handling
- Loading states during authentication

### Signup Page
- Complete registration form with name, email, and password
- Password strength indicator
- Password confirmation validation
- Terms of service agreement
- Social signup options
- Real-time form validation

### Dashboard Header
- Professional header with user profile
- User avatar and information display
- Dropdown menu with user options
- Notification and settings buttons
- Responsive design for mobile devices

## Components

### AuthContext (`src/contexts/AuthContext.tsx`)
- Manages authentication state throughout the application
- Provides login, signup, and logout functions
- Handles token storage and user session management
- Includes error handling and loading states

### Login Component (`src/components/Auth/Login.tsx`)
- Professional login form with modern UI
- Input validation and error display
- Loading states and disabled states
- Social login integration

### Signup Component (`src/components/Auth/Signup.tsx`)
- Complete registration form
- Password strength validation
- Real-time form validation
- Terms agreement checkbox

### AuthWrapper (`src/components/Auth/AuthWrapper.tsx`)
- Manages switching between login and signup forms
- Handles form state and error clearing
- Provides seamless user experience

### Header Component (`src/components/Dashboard/Header.tsx`)
- Professional dashboard header
- User profile dropdown menu
- Notification and settings buttons
- Responsive design

## Styling

### Auth.css (`src/components/Auth/Auth.css`)
- Modern, professional authentication styles
- Glass morphism effects
- Responsive design
- Smooth animations and transitions
- Professional color scheme

### Header.css (`src/components/Dashboard/Header.css`)
- Professional header styling
- User menu dropdown styles
- Responsive design
- Modern UI elements

## Usage

### Demo Credentials
For testing purposes, use these demo credentials:
- **Email:** demo@example.com
- **Password:** password

### Authentication Flow
1. User visits the application
2. If not authenticated, login page is displayed
3. User can switch between login and signup forms
4. After successful authentication, user is redirected to dashboard
5. Dashboard displays user information in header
6. User can logout via header dropdown menu

## Technical Implementation

### State Management
- Uses React Context for global authentication state
- Local storage for token persistence
- Automatic session restoration on page reload

### Security Features
- Password strength validation
- Form validation and sanitization
- Secure token storage
- Session management

### Responsive Design
- Mobile-first approach
- Adaptive layouts for different screen sizes
- Touch-friendly interface elements

## Future Enhancements

### Backend Integration
- Replace demo authentication with real backend API
- Implement JWT token validation
- Add password reset functionality
- Email verification system

### Additional Features
- Two-factor authentication
- Social login implementation
- User profile management
- Session timeout handling
- Remember me functionality

### Security Improvements
- CSRF protection
- Rate limiting
- Input sanitization
- Secure cookie handling

## File Structure

```
src/
├── contexts/
│   └── AuthContext.tsx
├── components/
│   ├── Auth/
│   │   ├── Login.tsx
│   │   ├── Signup.tsx
│   │   ├── AuthWrapper.tsx
│   │   └── Auth.css
│   └── Dashboard/
│       ├── Header.tsx
│       └── Header.css
└── App.tsx (updated with authentication)
```

## Getting Started

1. The authentication system is already integrated into the main App component
2. No additional setup required for demo mode
3. For production, replace the demo authentication logic with real backend calls
4. Customize styling by modifying the CSS files
5. Add additional features as needed

## Notes

- Current implementation uses demo authentication for development
- All authentication logic is centralized in AuthContext
- Styling follows modern design principles
- Components are fully responsive and accessible
- Error handling is comprehensive and user-friendly
