const express = require('express');
const router = express.Router();
const { registerUser, loginUser, getUserDetails } = require('../controllers/authController');
const { protect } = require('../middleware/authMiddleware');

// Register Route
router.post('/register-user', registerUser);

// Login Route
router.post('/login', loginUser);

// Protected Route to get User details
router.get('/get-userDetails', protect, getUserDetails);

module.exports = router;
