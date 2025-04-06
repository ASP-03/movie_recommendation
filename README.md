# Home Cinema - Movie Recommendation System

A sophisticated movie recommendation system that helps users discover similar movies based on their preferences. Built with Python, Flask, and modern web technologies.

<div align="center">
    <img src="https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-4.0.2&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1920&q=80" alt="Home Cinema Screenshot" width="800"/>
    <br>
    <h2>🎬 Home Cinema</h2>
</div>

## Features

- 🎬 **Intelligent Recommendations**: Uses advanced machine learning algorithms to find similar movies
- 🎨 **Modern UI**: Beautiful, responsive interface with smooth animations
- 🔍 **Smart Search**: Find movies by title with instant feedback
- 🏷️ **Genre Tags**: Color-coded genre tags for easy movie categorization
- ⭐ **Rating System**: Displays predicted ratings for recommended movies
- 📱 **Responsive Design**: Works seamlessly on all devices
- 🎯 **User-Friendly**: Intuitive interface with clear error handling

## Technologies Used

- **Backend**:
  - Python
  - Flask
  - Pandas
  - Scikit-learn
  - NumPy

- **Frontend**:
  - HTML5
  - CSS3
  - JavaScript
  - Bulma CSS Framework
  - Font Awesome
  - Animate.css
  - Toastify.js

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/home-cinema.git
   cd home-cinema
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Access the Application**
   Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter a movie title in the search box
2. Click "Find Similar Movies" or press Enter
3. View the recommendations with their ratings and genres
4. Click on genre tags to see more movies in that category

## Project Structure

```
home-cinema/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Main HTML template
├── static/
│   ├── css/           # Custom CSS files
│   └── js/            # JavaScript files
└── data/              # Movie dataset and model files
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Movie data sourced from [MovieLens](https://movielens.org/)
- Background image from [Unsplash](https://unsplash.com/)
- Icons from [Font Awesome](https://fontawesome.com/)
- CSS Framework by [Bulma](https://bulma.io/) 