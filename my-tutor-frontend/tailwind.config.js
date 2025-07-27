/** @type {import('tailwindcss').Config} */
module.exports = {
  // Specify the files where Tailwind should look for utility classes
  content: [
    "./src/**/*.{js,jsx,ts,tsx}", // Look in all JS/JSX/TS/TSX files inside the 'src' directory
    "./public/index.html",       // Also check your main HTML file
  ],
  theme: {
    extend: {
      // You can extend Tailwind's default theme here.
      // For example, adding custom colors, fonts, spacing, etc.
      fontFamily: {
        sans: ['Inter', 'sans-serif'], // Set 'Inter' as the default sans-serif font
      },
    },
  },
  plugins: [],
}