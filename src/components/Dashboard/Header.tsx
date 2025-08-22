import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import './Header.css';

interface HeaderProps {
  title: string;
  subtitle?: string;
}

const Header: React.FC<HeaderProps> = ({ title, subtitle }) => {
  const { user, logout } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);

  const handleLogout = () => {
    logout();
    setShowUserMenu(false);
  };

  return (
    <header className="dashboard-header">
      <div className="header-content">
        <div className="header-left">
                     <div className="logo-section">
             <div className="logo-icon">AI</div>
             <div className="logo-text">
               <h1>{title}</h1>
               {subtitle && <p>{subtitle}</p>}
             </div>
           </div>
        </div>

                 <div className="header-right">
           <div className="header-actions">
             <button className="action-button" title="Notifications">
               <span className="notification-badge">3</span>
             </button>
             
             <button className="action-button" title="Settings">
               Settings
             </button>

            <div className="user-profile">
              <button
                className="user-button"
                onClick={() => setShowUserMenu(!showUserMenu)}
                aria-expanded={showUserMenu}
              >
                <div className="user-avatar">
                  {user?.avatar ? (
                    <img src={user.avatar} alt={user.name} />
                  ) : (
                    <div className="avatar-placeholder">
                      {user?.name?.charAt(0).toUpperCase() || 'U'}
                    </div>
                  )}
                </div>
                <div className="user-info">
                  <span className="user-name">{user?.name || 'User'}</span>
                  <span className="user-email">{user?.email || 'user@example.com'}</span>
                </div>
                <span className="dropdown-icon">▼</span>
              </button>

              {showUserMenu && (
                <div className="user-menu">
                  <div className="menu-header">
                    <div className="menu-user-info">
                      <div className="menu-avatar">
                        {user?.avatar ? (
                          <img src={user.avatar} alt={user.name} />
                        ) : (
                          <div className="avatar-placeholder">
                            {user?.name?.charAt(0).toUpperCase() || 'U'}
                          </div>
                        )}
                      </div>
                      <div>
                        <div className="menu-user-name">{user?.name || 'User'}</div>
                        <div className="menu-user-email">{user?.email || 'user@example.com'}</div>
                      </div>
                    </div>
                  </div>
                  
                                     <div className="menu-items">
                     <button className="menu-item">
                       Profile Settings
                     </button>
                     <button className="menu-item">
                       Security
                     </button>
                     <button className="menu-item">
                       Analytics
                     </button>
                     <button className="menu-item">
                       Help & Support
                     </button>
                   </div>
                   
                   <div className="menu-divider"></div>
                   
                   <button className="menu-item logout" onClick={handleLogout}>
                     Sign Out
                   </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
