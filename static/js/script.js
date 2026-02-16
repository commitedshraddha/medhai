/**
 * MedhAI+ - Adaptive Learning System
 * JavaScript Utilities and Helper Functions
 * Handles API calls, dynamic content loading, and user interactions
 */

// ==================== UTILITY FUNCTIONS ====================

/**
 * Display a toast notification
 * @param {string} message - The message to display
 * @param {string} type - Type of notification (success, error, info, warning)
 */
function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Style the toast
    Object.assign(toast.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '1rem 1.5rem',
        borderRadius: '12px',
        backgroundColor: getToastColor(type),
        color: 'white',
        fontWeight: '600',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)',
        zIndex: '9999',
        animation: 'slideInRight 0.3s ease',
        maxWidth: '300px'
    });
    
    document.body.appendChild(toast);
    
    // Remove toast after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * Get color based on toast type
 * @param {string} type - Toast type
 * @returns {string} Color code
 */
function getToastColor(type) {
    const colors = {
        'success': '#27ae60',
        'error': '#e74c3c',
        'warning': '#f39c12',
        'info': '#3498db'
    };
    return colors[type] || colors.info;
}

/**
 * Format number to percentage
 * @param {number} value - Decimal value
 * @returns {string} Formatted percentage
 */
function formatPercentage(value) {
    return `${Math.round(value)}%`;
}

/**
 * Validate form input
 * @param {string} fieldName - Name of the field
 * @param {number} value - Value to validate
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {boolean} Whether the value is valid
 */
function validateInput(fieldName, value, min, max) {
    if (isNaN(value) || value === null || value === '') {
        showToast(`Please enter a valid ${fieldName}`, 'error');
        return false;
    }
    
    if (value < min || value > max) {
        showToast(`${fieldName} must be between ${min} and ${max}`, 'error');
        return false;
    }
    
    return true;
}

/**
 * Save data to localStorage
 * @param {string} key - Storage key
 * @param {*} data - Data to store
 */
function saveToStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
        console.error('Error saving to localStorage:', error);
    }
}

/**
 * Load data from localStorage
 * @param {string} key - Storage key
 * @returns {*} Stored data or null
 */
function loadFromStorage(key) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : null;
    } catch (error) {
        console.error('Error loading from localStorage:', error);
        return null;
    }
}

/**
 * Clear all localStorage data
 */
function clearStorage() {
    try {
        localStorage.clear();
        showToast('Storage cleared successfully', 'success');
    } catch (error) {
        console.error('Error clearing localStorage:', error);
    }
}

// ==================== API FUNCTIONS ====================

/**
 * Make a POST request to the server
 * @param {string} endpoint - API endpoint
 * @param {Object} data - Data to send
 * @returns {Promise<Object>} Response data
 */
async function postRequest(endpoint, data) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Request failed');
        }
        
        return result;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Make a GET request to the server
 * @param {string} endpoint - API endpoint
 * @returns {Promise<Object>} Response data
 */
async function getRequest(endpoint) {
    try {
        const response = await fetch(endpoint);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Request failed');
        }
        
        return result;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ==================== ANIMATION HELPERS ====================

/**
 * Animate element with fade in effect
 * @param {HTMLElement} element - Element to animate
 * @param {number} duration - Animation duration in ms
 */
function fadeIn(element, duration = 300) {
    element.style.opacity = '0';
    element.style.display = 'block';
    
    let opacity = 0;
    const increment = 50 / duration;
    
    const timer = setInterval(() => {
        opacity += increment;
        element.style.opacity = opacity.toString();
        
        if (opacity >= 1) {
            clearInterval(timer);
            element.style.opacity = '1';
        }
    }, 50);
}

/**
 * Animate element with fade out effect
 * @param {HTMLElement} element - Element to animate
 * @param {number} duration - Animation duration in ms
 */
function fadeOut(element, duration = 300) {
    let opacity = 1;
    const decrement = 50 / duration;
    
    const timer = setInterval(() => {
        opacity -= decrement;
        element.style.opacity = opacity.toString();
        
        if (opacity <= 0) {
            clearInterval(timer);
            element.style.opacity = '0';
            element.style.display = 'none';
        }
    }, 50);
}

/**
 * Smooth scroll to element
 * @param {string} selector - CSS selector
 */
function smoothScroll(selector) {
    const element = document.querySelector(selector);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// ==================== PROGRESS TRACKING ====================

/**
 * Track user progress
 */
class ProgressTracker {
    constructor() {
        this.storageKey = 'medhai_progress';
        this.progress = this.loadProgress();
    }
    
    loadProgress() {
        return loadFromStorage(this.storageKey) || {
            quizzes_taken: 0,
            total_score: 0,
            predictions: [],
            quiz_results: [],
            last_updated: null
        };
    }
    
    saveProgress() {
        this.progress.last_updated = new Date().toISOString();
        saveToStorage(this.storageKey, this.progress);
    }
    
    addPrediction(prediction) {
        this.progress.predictions.push({
            ...prediction,
            timestamp: new Date().toISOString()
        });
        this.saveProgress();
    }
    
    addQuizResult(result) {
        this.progress.quiz_results.push({
            ...result,
            timestamp: new Date().toISOString()
        });
        this.progress.quizzes_taken++;
        this.progress.total_score += result.score;
        this.saveProgress();
    }
    
    getAverageScore() {
        if (this.progress.quizzes_taken === 0) return 0;
        return this.progress.total_score / this.progress.quizzes_taken;
    }
    
    getLatestPrediction() {
        return this.progress.predictions[this.progress.predictions.length - 1] || null;
    }
    
    reset() {
        this.progress = {
            quizzes_taken: 0,
            total_score: 0,
            predictions: [],
            quiz_results: [],
            last_updated: null
        };
        this.saveProgress();
    }
}

// Create global progress tracker instance
const progressTracker = new ProgressTracker();

// ==================== FORM HELPERS ====================

/**
 * Disable form during submission
 * @param {HTMLFormElement} form - Form element
 */
function disableForm(form) {
    const inputs = form.querySelectorAll('input, button, textarea, select');
    inputs.forEach(input => {
        input.disabled = true;
        input.style.opacity = '0.6';
    });
}

/**
 * Enable form after submission
 * @param {HTMLFormElement} form - Form element
 */
function enableForm(form) {
    const inputs = form.querySelectorAll('input, button, textarea, select');
    inputs.forEach(input => {
        input.disabled = false;
        input.style.opacity = '1';
    });
}

/**
 * Collect form data as object
 * @param {HTMLFormElement} form - Form element
 * @returns {Object} Form data
 */
function getFormData(form) {
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        // Try to convert to number if possible
        const numValue = parseFloat(value);
        data[key] = isNaN(numValue) ? value : numValue;
    }
    
    return data;
}

// ==================== INITIALIZATION ====================

/**
 * Initialize app on DOM load
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('MedhAI+ Application Initialized');
    
    // Add smooth scroll behavior to all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add animation styles dynamically
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
});

// ==================== EXPORT FUNCTIONS ====================

// Make functions available globally
window.MedhAI = {
    showToast,
    formatPercentage,
    validateInput,
    saveToStorage,
    loadFromStorage,
    clearStorage,
    postRequest,
    getRequest,
    fadeIn,
    fadeOut,
    smoothScroll,
    progressTracker,
    disableForm,
    enableForm,
    getFormData
};

console.log('MedhAI+ JavaScript loaded successfully!');
