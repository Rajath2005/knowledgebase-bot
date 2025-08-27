# knowledgebase-bot
A customizable AI-powered chatbot designed to answer queries from a knowledge base. Built for experimenting with chat interfaces, document-based Q&amp;A, and integration with open-source models.

# Overview

This is a Flask-based chatbot application that integrates with Hugging Face's BlenderBot API to provide conversational AI capabilities. The application serves as a lightweight web service that accepts user messages and returns AI-generated responses using Facebook's BlenderBot-400M-distill model through Hugging Face's inference API.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Architecture
- **Framework**: Flask web framework for creating a lightweight REST API
- **API Integration**: Direct HTTP client integration with Hugging Face's inference API
- **Error Handling**: Structured error handling with JSON responses for API failures
- **Logging**: Debug-level logging throughout the application for troubleshooting

## API Design
- **RESTful Structure**: Flask routes handle HTTP requests and return JSON responses
- **Request Processing**: Accepts user messages and forwards them to the Hugging Face API
- **Response Processing**: Processes and formats responses from the AI model before returning to clients

## Configuration Management
- **Environment Variables**: Uses environment variables for sensitive configuration like API keys
- **API Configuration**: Configurable Hugging Face API endpoint and authentication

## AI Model Integration
- **Model Choice**: Facebook's BlenderBot-400M-distill model via Hugging Face
- **Inference Method**: Uses Hugging Face's hosted inference API rather than local model execution
- **Request Options**: Includes "wait_for_model" option to handle model loading delays

# External Dependencies

## Third-Party APIs
- **Hugging Face Inference API**: Primary dependency for AI conversation capabilities
- **BlenderBot Model**: Facebook's conversational AI model hosted on Hugging Face

## Python Libraries
- **Flask**: Web framework for HTTP server functionality
- **Requests**: HTTP client library for external API communication
- **OS**: Environment variable access for configuration
- **JSON**: Data serialization for API communication
- **Logging**: Application logging and debugging

## Environment Configuration
- **HUGGINGFACE_API_KEY**: Required environment variable for API authentication
- **API Timeout**: 30-second timeout configured for external API calls