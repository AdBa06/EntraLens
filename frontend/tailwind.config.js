/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}', '../backend/templates/**/*'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0078D4',
          dark: '#005A9E',
          light: '#2B88D8',
        },
        secondary: {
          DEFAULT: '#2B88D8',
          dark: '#005A9E',
          light: '#6CB8F6',
        },
        accent: '#107C10',
      },
    },
  },
  plugins: [],
};
