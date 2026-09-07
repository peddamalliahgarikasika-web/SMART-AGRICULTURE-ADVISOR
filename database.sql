CREATE DATABASE smart_agriculture;

USE smart_agriculture;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    password VARCHAR(255)
);

CREATE TABLE crops (
    id INT AUTO_INCREMENT PRIMARY KEY,
    crop_name VARCHAR(100),
    soil_type VARCHAR(100),
    season VARCHAR(100),
    recommendation TEXT
);

CREATE TABLE fertilizer (
    id INT AUTO_INCREMENT PRIMARY KEY,
    crop_name VARCHAR(100),
    fertilizer_name VARCHAR(100),
    quantity VARCHAR(100)
);

CREATE TABLE disease_reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    farmer_name VARCHAR(100),
    disease_name VARCHAR(100),
    image_path VARCHAR(255)
);

CREATE TABLE feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    message TEXT
);

INSERT INTO crops(crop_name, soil_type, season, recommendation)
VALUES
('Rice', 'Clay', 'Monsoon', 'Rice grows best in clay soil during monsoon'),
('Wheat', 'Loamy', 'Winter', 'Wheat is suitable for cool climate'),
('Cotton', 'Black Soil', 'Summer', 'Cotton grows well in black soil');

INSERT INTO fertilizer(crop_name, fertilizer_name, quantity)
VALUES
('Rice', 'Urea', '50kg per acre'),
('Wheat', 'DAP', '40kg per acre'),
('Cotton', 'Potash', '30kg per acre');