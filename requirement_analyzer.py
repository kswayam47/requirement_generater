import google.generativeai as genai
import os
from typing import List
import json
from dotenv import load_dotenv
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequirementsAnalyzer:
    def __init__(self):
        """Initialize the RequirementsAnalyzer"""
        logger.info("Initializing RequirementsAnalyzer")
        load_dotenv()
        
        # Try multiple environment variable names for API key
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.error("Error: No API key found in environment")
            raise ValueError("No API key found in environment. Set GEMINI_API_KEY or GOOGLE_API_KEY")
            
        logger.info("Configuring Gemini API")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.chat = None
        
        # Initialize categories and cache
        self.question_categories = [
            "Functional Requirements",
            "Non-Functional Requirements",
            "User Interface & UX",
            "Security & Privacy",
            "Data Management & Storage",
            "System Integration",
            "Performance & Scalability",
            "Stakeholder Requirements",
            "Business Rules & Workflows",
            "Technical Architecture",
            "Deployment & Infrastructure",
            "Compliance & Standards",
            "Error Handling & Recovery",
            "Monitoring & Logging",
            "Documentation Requirements",
            "Testing Requirements",
            "Maintenance & Support"
        ]
        self.response_cache = {}
        self.cache_file = "requirements_cache.json"
        self.load_cache()

    def load_cache(self):
        """Load cached responses from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.response_cache = json.load(f)
            else:
                self.response_cache = {}
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self.response_cache = {}

    def save_cache(self):
        """Save cached responses to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.response_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def get_cache_key(self, text: str) -> str:
        """Generate a cache key from text by removing whitespace and converting to lowercase"""
        return ' '.join(text.lower().split())

    def get_cached_response(self, prompt: str) -> str:
        """Get response from cache if it exists"""
        cache_key = self.get_cache_key(prompt)
        return self.response_cache.get(cache_key)

    def cache_response(self, prompt: str, response: str):
        """Cache a response"""
        cache_key = self.get_cache_key(prompt)
        self.response_cache[cache_key] = response
        self.save_cache()

    def setup_api(self, api_key: str):
        """Setup the Gemini API with the provided key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')

    def generate_analysis_prompt(self, initial_prompt: str) -> str:
        """Generate the analysis prompt for the AI"""
        return f"""Analyze this project description and identify domain-specific terminology first:

        {initial_prompt}

        1. Identify the system domain and key roles/terms
        2. Then analyze requirements using domain-appropriate terminology

        Format your response as a JSON object:
        {{
            "domain_info": {{
                "name": "identified_domain",
                "primary_users": ["list of main user types in this domain"],
                "secondary_users": ["list of other user types"],
                "admin_users": ["list of administrative user types"],
                "key_terms": {{
                    "generic_term": "domain_specific_term"
                }}
            }},
            "srs_basics": {{
                "purpose": [
                    {{
                        "question": "What is the primary purpose of this document?",
                        "context": "Define the intended use and audience of the SRS",
                        "options": [
                            "Technical implementation guide for development team",
                            "Project planning and estimation reference",
                            "Contractual agreement with stakeholders",
                            "Comprehensive system documentation for all parties"
                        ]
                    }}
                ],
                "scope": [
                    {{
                        "question": "What is the scope of this system?",
                        "context": "Define system boundaries and what is in/out of scope",
                        "options": [
                            "Core System Scope: Focus on essential features and core functionality only. Includes basic user management, data processing, and primary business logic.",
                            "Comprehensive System Scope: Full implementation of all features including advanced functionality, third-party integrations, and administrative tools.",
                            "Extended Enterprise Scope: Enterprise-wide implementation including core features, all integrations, advanced analytics, and API access."
                        ]
                    }}
                ],
                "stakeholders": [
                    {{
                        "question": "Who are the key stakeholders and user classes?",
                        "context": "Define all parties involved in or affected by the system",
                        "dynamic_options": true
                    }}
                ],
                "constraints": [
                    {{
                        "question": "What are the technical constraints?",
                        "context": "Define technical limitations and requirements",
                        "options": [
                            "Basic Technical Stack: Must use existing infrastructure. Limited to approved enterprise technologies.",
                            "Flexible Technical Stack: Can introduce new technologies with approval. Cloud-native architecture.",
                            "Advanced Technical Stack: Full freedom in technology choice. Hybrid cloud architecture."
                        ]
                    }},
                    {{
                        "question": "What are the operational constraints?",
                        "context": "Define operational limitations and requirements",
                        "options": [
                            "Standard Operations: Business hours support. Monthly updates. 99% uptime SLA.",
                            "Enhanced Operations: Extended hours support. Weekly updates. 99.9% uptime SLA.",
                            "Enterprise Operations: 24/7 support. Continuous updates. 99.99% uptime SLA."
                        ]
                    }}
                ],
                "assumptions": [
                    {{
                        "question": "What are the technical assumptions?",
                        "context": "Define assumed technical conditions",
                        "options": [
                            "Basic Infrastructure: Existing hardware and network infrastructure will be sufficient. Standard internet connectivity available.",
                            "Modern Infrastructure: Cloud resources available. High-speed internet connectivity. Modern browser support required.",
                            "Advanced Infrastructure: Dedicated cloud resources. Multi-region deployment capability. Latest technology support."
                        ]
                    }},
                    {{
                        "question": "What are the organizational assumptions?",
                        "context": "Define assumed organizational conditions",
                        "options": [
                            "Minimal Organization: Small team with basic roles. Limited stakeholder involvement. Standard project management.",
                            "Standard Organization: Dedicated team with defined roles. Regular stakeholder engagement. Agile project management.",
                            "Enterprise Organization: Multiple teams with specialized roles. Full stakeholder participation. Advanced project management."
                        ]
                    }},
                    {{
                        "question": "What are the user assumptions?",
                        "context": "Define assumed user conditions",
                        "options": [
                            "Basic Users: Technical literacy required. English-only interface. Desktop-primary usage.",
                            "Standard Users: Mixed technical expertise. Multi-language support. Multi-device usage.",
                            "Advanced Users: High technical proficiency. Full internationalization. All device types supported."
                        ]
                    }}
                ]
            }},
            "known_requirements": [
                "List each requirement that is explicitly stated or can be clearly derived"
            ],
            "questions": {{
                "Functional Requirements": [
                    {{
                        "question": "Detailed question about specific functionality",
                        "context": "Brief explanation of why this is important",
                        "options": [
                            "Detailed option 1 with specific implementation details",
                            "Detailed option 2 with specific implementation details",
                            "Detailed option 3 with specific implementation details"
                        ]
                    }}
                ],
                "Non-Functional Requirements": [
                    {{
                        "question": "Specific question about system quality attributes",
                        "context": "Technical impact of this requirement",
                        "options": [
                            "Specific measurable target 1",
                            "Specific measurable target 2",
                            "Specific measurable target 3"
                        ]
                    }}
                ]
            }},
            "innovative_features": [
                {{
                    "feature_name": "Name of innovative feature",
                    "description": "Detailed description of the feature and its benefits",
                    "follow_up_questions": [
                        {{
                            "question": "Specific implementation question",
                            "options": [
                                "Detailed implementation option 1",
                                "Detailed implementation option 2",
                                "Detailed implementation option 3"
                            ]
                        }}
                    ]
                }}
            ]
        }}

        IMPORTANT GUIDELINES:
        1. SRS Basics:
        - Focus on clear project boundaries
        - Include all necessary stakeholders
        - Define clear assumptions and constraints
        - Identify external dependencies
        - Consider regulatory requirements

        2. Requirements:
        - Be specific and measurable
        - Include technical details
        - Consider system boundaries
        - Address quality attributes
        - Include acceptance criteria

        3. Features:
        - Align with project goals
        - Consider technical feasibility
        - Address user needs
        - Include implementation details
        """
        return f"""Analyze the following project description and create a comprehensive Software Requirements Specification (SRS) following IEEE 830 format:

        {initial_prompt}

        First, gather all essential SRS document information, then analyze requirements. Structure the response as follows:

        1. Introduction and Project Overview:
        - Purpose and intended audience
        - Project scope and objectives
        - Stakeholders and user classes
        - System context and environment
        - Constraints and assumptions
        - Dependencies and prerequisites

        2. References and Definitions:
        - Related documents and standards
        - Technical terms and acronyms
        - External system interfaces

        3. System Features and Requirements Analysis:
        a) Identify explicitly stated requirements
        b) Generate questions for unclear/missing requirements
        c) Suggest innovative features

        Format your response as a JSON object with this exact structure:
        {{
            "domain_info": {{
                "name": "identified_domain",
                "primary_users": ["list of main user types in this domain"],
                "secondary_users": ["list of other user types"],
                "admin_users": ["list of administrative user types"],
                "key_terms": {{
                    "generic_term": "domain_specific_term"
                }}
            }},
            "srs_basics": {{
                "purpose": [
                    {{
                        "question": "What is the primary purpose of this document?",
                        "context": "Define the intended use and audience of the SRS",
                        "options": [
                            "Technical implementation guide for development team",
                            "Project planning and estimation reference",
                            "Contractual agreement with stakeholders",
                            "Comprehensive system documentation for all parties"
                        ]
                    }}
                ],
                "scope": [
                    {{
                        "question": "What is the scope of this system?",
                        "context": "Define system boundaries and what is in/out of scope",
                        "options": [
                            "Core System Scope: Focus on essential features and core functionality only. Includes basic user management, data processing, and primary business logic.",
                            "Comprehensive System Scope: Full implementation of all features including advanced functionality, third-party integrations, and administrative tools.",
                            "Extended Enterprise Scope: Enterprise-wide implementation including core features, all integrations, advanced analytics, and API access."
                        ]
                    }}
                ],
                "stakeholders": [
                    {{
                        "question": "Who are the key stakeholders and user classes?",
                        "context": "Define all parties involved in or affected by the system",
                        "dynamic_options": true
                    }}
                ],
                "constraints": [
                    {{
                        "question": "What are the technical constraints?",
                        "context": "Define technical limitations and requirements",
                        "options": [
                            "Basic Technical Stack: Must use existing infrastructure. Limited to approved enterprise technologies.",
                            "Flexible Technical Stack: Can introduce new technologies with approval. Cloud-native architecture.",
                            "Advanced Technical Stack: Full freedom in technology choice. Hybrid cloud architecture."
                        ]
                    }},
                    {{
                        "question": "What are the operational constraints?",
                        "context": "Define operational limitations and requirements",
                        "options": [
                            "Standard Operations: Business hours support. Monthly updates. 99% uptime SLA.",
                            "Enhanced Operations: Extended hours support. Weekly updates. 99.9% uptime SLA.",
                            "Enterprise Operations: 24/7 support. Continuous updates. 99.99% uptime SLA."
                        ]
                    }}
                ],
                "assumptions": [
                    {{
                        "question": "What are the technical assumptions?",
                        "context": "Define assumed technical conditions",
                        "options": [
                            "Basic Infrastructure: Existing hardware and network infrastructure will be sufficient. Standard internet connectivity available.",
                            "Modern Infrastructure: Cloud resources available. High-speed internet connectivity. Modern browser support required.",
                            "Advanced Infrastructure: Dedicated cloud resources. Multi-region deployment capability. Latest technology support."
                        ]
                    }},
                    {{
                        "question": "What are the organizational assumptions?",
                        "context": "Define assumed organizational conditions",
                        "options": [
                            "Minimal Organization: Small team with basic roles. Limited stakeholder involvement. Standard project management.",
                            "Standard Organization: Dedicated team with defined roles. Regular stakeholder engagement. Agile project management.",
                            "Enterprise Organization: Multiple teams with specialized roles. Full stakeholder participation. Advanced project management."
                        ]
                    }},
                    {{
                        "question": "What are the user assumptions?",
                        "context": "Define assumed user conditions",
                        "options": [
                            "Basic Users: Technical literacy required. English-only interface. Desktop-primary usage.",
                            "Standard Users: Mixed technical expertise. Multi-language support. Multi-device usage.",
                            "Advanced Users: High technical proficiency. Full internationalization. All device types supported."
                        ]
                    }}
                ]
            }},
            "known_requirements": [
                "List each requirement that is explicitly stated or can be clearly derived"
            ],
            "questions": {{
                "Functional Requirements": [
                    {{
                        "question": "Detailed question about specific functionality",
                        "context": "Brief explanation of why this is important",
                        "options": [
                            "Detailed option 1 with specific implementation details",
                            "Detailed option 2 with specific implementation details",
                            "Detailed option 3 with specific implementation details"
                        ]
                    }}
                ],
                "Non-Functional Requirements": [
                    {{
                        "question": "Specific question about system quality attributes",
                        "context": "Technical impact of this requirement",
                        "options": [
                            "Specific measurable target 1",
                            "Specific measurable target 2",
                            "Specific measurable target 3"
                        ]
                    }}
                ]
            }},
            "innovative_features": [
                {{
                    "feature_name": "Name of innovative feature",
                    "description": "Detailed description of the feature and its benefits",
                    "follow_up_questions": [
                        {{
                            "question": "Specific implementation question",
                            "options": [
                                "Detailed implementation option 1",
                                "Detailed implementation option 2",
                                "Detailed implementation option 3"
                            ]
                        }}
                    ]
                }}
            ]
        }}

        IMPORTANT GUIDELINES:
        1. SRS Basics:
        - Focus on clear project boundaries
        - Include all necessary stakeholders
        - Define clear assumptions and constraints
        - Identify external dependencies
        - Consider regulatory requirements

        2. Requirements:
        - Be specific and measurable
        - Include technical details
        - Consider system boundaries
        - Address quality attributes
        - Include acceptance criteria

        3. Features:
        - Align with project goals
        - Consider technical feasibility
        - Address user needs
        - Include implementation details
        """
        return base_prompt

    def convert_to_statement(self, question: str, answer: str) -> str:
        """Convert a Q&A pair into a natural statement"""
        # Clean up the question text
        question = question.strip('?').lower()
        
        patterns = {
            "what type": "The type of",
            "what kind": "The kind of",
            "what level": "The level of",
            "what are the": "The",
            "what is the": "The",
            "how should": "The system will",
            "how many": "The number of",
            "should the system": "The system will",
            "which": "The selected",
            "where": "The location for",
            "when": "The timing for"
        }
        
        for pattern, replacement in patterns.items():
            if question.startswith(pattern):
                question = question.replace(pattern, replacement, 1)
                break
        
      
        if "wearable" in question:
            return f"Supported wearable devices and data types: {answer}"
        elif "security" in question:
            return f"Security requirements: {answer}"
        elif "performance" in question:
            return f"Performance requirements: {answer}"
        elif "integration" in question:
            return f"Integration specifications: {answer}"
        elif "database" in question or "data storage" in question:
            return f"Data storage solution: {answer}"
        elif "authentication" in question:
            return f"Authentication method: {answer}"
        elif "backup" in question:
            return f"Backup strategy: {answer}"
        elif "monitoring" in question:
            return f"Monitoring approach: {answer}"
        elif "notification" in question or "alert" in question:
            return f"Notification system: {answer}"
        elif "user interface" in question or "ui" in question:
            return f"User interface implementation: {answer}"
        elif "api" in question:
            return f"API specifications: {answer}"
        elif "compliance" in question:
            return f"Compliance requirements: {answer}"
        elif "scalability" in question:
            return f"Scalability approach: {answer}"
        elif "availability" in question:
            return f"Availability requirements: {answer}"
        elif "disaster recovery" in question:
            return f"Disaster recovery plan: {answer}"
        elif "logging" in question:
            return f"Logging requirements: {answer}"
        elif "reporting" in question:
            return f"Reporting capabilities: {answer}"
        elif "role" in question or "permission" in question:
            return f"User roles and permissions: {answer}"
        elif "workflow" in question or "process" in question:
            return f"Process workflow: {answer}"
        elif "validation" in question:
            return f"Validation rules: {answer}"
        elif "error handling" in question:
            return f"Error handling approach: {answer}"
        
        
        for word in ['what', 'how', 'are', 'is', 'will', 'should', 'can', 'does', 'do', 'please', 'specify']:
            if question.startswith(word + " "):
                question = question[len(word):].strip()
        
      
        question = question.strip()
        if not question:
            return f"Requirement: {answer}"
            
        return f"{question.capitalize()}: {answer}"

    def extract_json_from_response(self, response_text: str) -> dict:
        """Helper method to extract JSON from Gemini API response"""
        try:
            # Find the first { or [ and last } or ]
            json_start = min((response_text.find('{'), response_text.find('[')))
            if json_start == -1:  # If no { found, try just [
                json_start = response_text.find('[')
            if json_start == -1:  # If no [ found, try just {
                json_start = response_text.find('{')
            if json_start == -1:
                raise ValueError("No JSON start found")
            
            # Find matching end
            start_char = response_text[json_start]
            end_char = '}' if start_char == '{' else ']'
            json_end = response_text.rfind(end_char) + 1
            if json_end <= 0:
                raise ValueError("No JSON end found")
            
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            logger.error(f"Response text: {response_text}")
            return {} if start_char == '{' else []

    def detect_requirement_gaps(self, project_context: str, known_requirements: list) -> list:
        """Analyze the project for missing critical requirements"""
        gap_detection_prompt = f"""Analyze this healthcare system project and identify critical missing requirements that should be present but are not mentioned in the known requirements. Focus on industry standards, security, compliance, and essential features.

        Project Context:
        {project_context}

        Known Requirements:
        {json.dumps(known_requirements, indent=2)}

        Identify gaps in these categories:
        1. Core System Features
        2. Security & Privacy
        3. User Management & Access Control
        4. Data Management
        5. Integration & Interoperability
        6. Compliance & Auditing
        7. Performance & Reliability
        8. User Experience
        9. Administration & Monitoring
        10. Business Operations

        For each gap:
        1. Clearly explain why it's critical
        2. Reference industry standards or best practices
        3. Explain potential risks of omitting it

        Format as JSON array:
        [
            {{
                "category": "Category name",
                "gap": "Description of the missing requirement",
                "criticality": "High/Medium/Low",
                "justification": "Why this is critical",
                "industry_standard": "Relevant standard or best practice",
                "risk": "Risk of not implementing"
            }}
        ]
        """

        try:
            gap_response = self.model.generate_content(gap_detection_prompt)
            return self.extract_json_from_response(gap_response.text)
        except Exception as e:
            logger.error(f"Error detecting gaps: {str(e)}")
            return []

    def extract_explicit_requirements(self, project_context: str) -> dict:
        """Step 1: Extract and categorize explicit requirements from the problem statement"""
        try:
            prompt = f"""Analyze this project description and extract explicit requirements:

{project_context}

First, identify the system domain. Then categorize requirements using domain-specific categories.
DO NOT use generic medical/healthcare categories unless the project is specifically in that domain.

Return a JSON object:
{{
    "domain_info": {{
        "name": "identified_domain_name",
        "description": "brief_domain_description"
    }},
    "categories": [
        {{
            "name": "domain_specific_category_name",
            "requirements": [
                {{
                    "requirement": "specific_requirement_text",
                    "type": "Functional or Non-Functional",
                    "source": "relevant_part_of_input_text"
                }}
            ]
        }}
    ]
}}

For a banking system example, use categories like:
- Account Management
- Transaction Processing
- Customer Service
- Security & Compliance
- System Administration
- Financial Operations
- Integration & APIs

For other domains, use appropriate domain-specific categories.
DO NOT include categories that don't match the identified domain."""

            response = self.model.generate_content(prompt)
            result = self.extract_json_from_response(response.text)
            
            # Validate domain info
            if not isinstance(result.get("domain_info"), dict):
                raise ValueError("Missing or invalid domain_info")
            
            # Validate categories
            categories = result.get("categories", [])
            if not isinstance(categories, list):
                raise ValueError("Invalid categories format")
            
            # Validate each category and its requirements
            for category in categories:
                if not isinstance(category, dict) or "name" not in category or "requirements" not in category:
                    raise ValueError("Invalid category structure")
                if not isinstance(category["requirements"], list):
                    raise ValueError("Invalid requirements format")
                for req in category["requirements"]:
                    if not all(key in req for key in ["requirement", "type", "source"]):
                        raise ValueError("Invalid requirement structure")
                    if req["type"] not in ["Functional", "Non-Functional"]:
                        raise ValueError("Invalid requirement type")

            return result
        except Exception as e:
            logger.error(f"Error extracting requirements: {str(e)}")
            return {"domain_info": {}, "categories": []}

    def perform_gap_analysis(self, project_context: str, known_requirements: dict) -> list:
        """Step 2: Identify missing requirements based on industry standards"""
        checklist_prompt = f"""Compare these requirements against standard telemedicine system features.
        
        Project Context:
        {project_context}
        
        Current Requirements:
        {json.dumps(known_requirements, indent=2)}
        
        Check against these categories:
        1. Users Management
        2. Administrators Management
        3. Appointment System
        4. Video Consultation
        5. Prescription Management
        6. Medical Records
        7. Payment Processing
        8. Notifications
        9. Security & Privacy
        10. Performance & Scalability
        11. Integration & APIs
        12. Compliance & Auditing
        13. User Experience
        14. Administrative Features
        15. Reporting & Analytics

        For each gap found, provide:
        1. Category it belongs to
        2. What's missing
        3. Why it's important
        4. Industry standard or best practice reference
        5. Risk of not including it
        6. Suggested implementation approach

        Format as JSON array:
        [
            {{
                "category": "Category name",
                "gap": "Missing requirement",
                "importance": "Why it's critical",
                "standard": "Industry standard reference",
                "risk": "Risk of omission",
                "suggestion": "Implementation suggestion",
                "priority": "High/Medium/Low"
            }}
        ]
        """
        try:
            response = self.model.generate_content(checklist_prompt)
            return self.extract_json_from_response(response.text)
        except Exception as e:
            logger.error(f"Error in gap analysis: {str(e)}")
            return []

    def generate_gap_questions(self, gaps: list) -> list:
        """Step 3: Generate questions to address identified gaps"""
        questions = []
        for gap in gaps:
            question = {
                "category": gap["category"],
                "question": f"Regarding {gap['gap'].lower()}, which approach would you prefer?",
                "context": f"This is important because: {gap['importance']}\nIndustry standard: {gap['standard']}\nRisk: {gap['risk']}",
                "options": [
                    gap["suggestion"],
                    f"Alternative approach to {gap['gap']}",
                    "Custom implementation",
                    "Not required for initial release"
                ]
            }
            questions.append(question)
        return questions

    def generate_clarifying_questions(self, project_context: str, explicit_reqs: dict, gap_responses: dict) -> list:
        """Step 4: Generate clarifying questions based on requirements and gaps"""
        try:
            # Define the standard categories
            categories = {
                "functional": "Core Business Logic",
                "non_functional": [
                    "Performance",
                    "Scalability", 
                    "Reliability",
                    "Availability",
                    "Security",
                    "Maintainability",
                    "Portability",
                    "Usability",
                    "Compatibility",
                    "Documentation",
                    "Compliance"
                ]
            }

            prompt = f"""Based on this project context and domain:
{project_context}

And these existing requirements (DO NOT ask about these again):
{json.dumps(explicit_reqs, indent=2)}

Generate domain-specific clarifying questions following these rules:
1. Focus ONLY on requirements not already covered in the existing requirements
2. For Functional category "{categories['functional']}", ask about core business rules and workflows
3. For each Non-Functional category {json.dumps(categories['non_functional'])}, ask specific technical questions
4. Use domain-specific terminology for the domain identified in the context
5. Each question MUST have exactly 3 clear, distinct options
6. Options should be specific, measurable, and relevant to the domain

Return a JSON array of categories:
[
    {{
        "aspect": "Category name from the provided categories list",
        "questions": [
            {{
                "question": "Specific question about uncovered requirements",
                "context": "Why this matters for this specific domain",
                "impact": "Technical/business impact of this decision",
                "options": [
                    "Detailed option 1 with specific metrics or criteria",
                    "Detailed option 2 with specific metrics or criteria",
                    "Detailed option 3 with specific metrics or criteria"
                ]
            }}
        ]
    }}
]"""

            response = self.model.generate_content(prompt)
            questions = self.extract_json_from_response(response.text)
            
            # Validate the response format
            if not isinstance(questions, list):
                raise ValueError("Invalid response format - expected list")
            
            for category in questions:
                if not isinstance(category, dict) or 'aspect' not in category or 'questions' not in category:
                    raise ValueError("Invalid category format")
                for q in category['questions']:
                    if not isinstance(q, dict) or 'question' not in q or 'options' not in q:
                        raise ValueError("Invalid question format")
                    if not isinstance(q['options'], list) or len(q['options']) != 3:
                        raise ValueError("Each question must have exactly 3 options")

            return questions

        except Exception as e:
            logger.error(f"Error generating clarifying questions: {str(e)}")
            return []

    def suggest_innovative_features(self, all_requirements: dict) -> dict:
        """Step 6: Suggest additional innovative features and get user preferences"""
        try:
            # Extract domain and key requirements for context
            domain_info = all_requirements.get("explicit_requirements", {}).get("domain_info", {})
            domain = domain_info.get("name", "general")
            
            prompt = f"""Based on these requirements for a {domain} system:
{json.dumps(all_requirements, indent=2)}

Generate innovative features that enhance the system's value. Follow these rules:
1. Each feature must be relevant to the {domain} domain
2. Features must go beyond basic requirements
3. Features must leverage modern technology trends
4. Features must be technically feasible
5. Each feature must have clear configuration options

Return a JSON array of features:
{{
    "innovative_features": [
        {{
            "name": "Feature name using domain terminology",
            "description": "Detailed explanation of the feature and its benefits",
            "impact": "Business and technical impact of this feature",
            "estimated_effort": "Implementation effort (Low/Medium/High)",
            "prerequisites": ["List of technical prerequisites"],
            "configuration_questions": [
                {{
                    "question": "Specific configuration question",
                    "options": ["Option 1", "Option 2", "Option 3"],
                    "impact": "How this choice affects implementation"
                }}
            ]
        }}
    ]
}}"""

            response = self.model.generate_content(prompt)
            result = self.extract_json_from_response(response.text)
            
            if not isinstance(result, dict) or "innovative_features" not in result:
                raise ValueError("Invalid response format - missing innovative_features")
            
            features = result["innovative_features"]
            if not isinstance(features, list) or not features:
                raise ValueError("Invalid features format or empty features list")
            
            # Ask user about each feature
            selected_features = []
            feature_requirements = {}
            
            print("\nProposed Innovative Features:")
            print("============================")
            
            for idx, feature in enumerate(features, 1):
                print(f"\nFeature {idx}: {feature['name']}")
                print(f"Description: {feature['description']}")
                print(f"Impact: {feature['impact']}")
                print(f"Estimated Effort: {feature['estimated_effort']}")
                print("\nPrerequisites:")
                for prereq in feature['prerequisites']:
                    print(f"- {prereq}")
                
                while True:
                    choice = input("\nWould you like to include this feature? (yes/no): ").strip().lower()
                    if choice in ['yes', 'no']:
                        break
                    print("Please enter 'yes' or 'no'")
                
                if choice == 'yes':
                    selected_features.append(feature)
                    feature_reqs = []
                    
                    print("\nConfiguration Questions:")
                    for q in feature['configuration_questions']:
                        print(f"\nQuestion: {q['question']}")
                        print(f"Impact: {q['impact']}")
                        print("\nOptions:")
                        for i, opt in enumerate(q['options'], 1):
                            print(f"{i}. {opt}")
                        
                        while True:
                            opt_choice = input("\nEnter option number: ").strip()
                            if opt_choice.isdigit() and 1 <= int(opt_choice) <= len(q['options']):
                                feature_reqs.append({
                                    "question": q['question'],
                                    "selected_option": q['options'][int(opt_choice) - 1]
                                })
                                break
                            print("Please enter a valid option number.")
                    feature_requirements[feature['name']] = feature_reqs

            return {
                "selected_features": selected_features,
                "feature_requirements": feature_requirements
            }

        except Exception as e:
            logger.error(f"Error suggesting innovative features: {str(e)}")
            return {"selected_features": [], "feature_requirements": {}}

    def auto_categorize_requirement(self, req: dict) -> str:
        """Automatically categorize a requirement using predefined rules"""
        # Extract the requirement text and other metadata
        if isinstance(req, dict):
            # Handle different requirement formats
            if "statement" in req:
                requirement_text = str(req["statement"]).lower()
                priority = str(req.get("priority", "")).lower()
                category = str(req.get("category", "")).lower()
            else:
                requirement_text = str(req.get("requirement", "")).lower()
                priority = str(req.get("priority", "")).lower()
                category = str(req.get("category", "")).lower()
        else:
            requirement_text = str(req).lower()
            priority = ""
            category = ""
        
        # Must Have - Critical system functionality and security
        must_keywords = [
            "security", "authentication", "encryption", "critical", "must", "shall", 
            "required", "essential", "core", "login", "access control", "audit",
            "compliance", "pci", "gdpr", "high priority", "data protection"
        ]
        
        # Should Have - Important business requirements
        should_keywords = [
            "should", "important", "performance", "monitoring", "reporting", 
            "backup", "recovery", "notification", "alert", "medium priority",
            "transaction", "account management", "customer service"
        ]
        
        # Could Have - Enhancement features
        could_keywords = [
            "could", "nice to have", "enhancement", "additional", "optional", 
            "improve", "analytics", "dashboard", "low priority", "reporting",
            "convenience", "user experience"
        ]

        # Automatic categorization logic
        if any(keyword in requirement_text for keyword in must_keywords):
            return "must"
        elif "security" in category or "compliance" in category:
            return "must"
        elif "high" in priority:
            return "must"
        elif any(keyword in requirement_text for keyword in should_keywords):
            return "should"
        elif "medium" in priority:
            return "should"
        elif any(keyword in requirement_text for keyword in could_keywords):
            return "could"
        elif "low" in priority:
            return "could"
        else:
            return "should"

    def get_gemini_priority(self, requirement: dict) -> str:
        """Use Gemini to determine the priority of a requirement using MoSCoW method"""
        try:
            # Create a detailed prompt for Gemini
            prompt = f"""As a banking system requirements expert, analyze this requirement and categorize it using the MoSCoW method (MUST, SHOULD, COULD, WON'T).

Requirement: {requirement.get('statement', requirement.get('requirement', ''))}
Type: {requirement.get('type', 'Not specified')}
Category: {requirement.get('category', 'Not specified')}
Priority: {requirement.get('priority', 'Not specified')}

Consider these factors:
1. Business Value: How critical is this to core banking operations?
2. Legal/Security Impact: Does this affect compliance, security, or risk management?
3. Dependencies: Is this a prerequisite for other features?
4. Implementation Complexity: What's the effort vs. benefit ratio?

Respond with ONLY ONE of these exact words: MUST, SHOULD, COULD, or WONT"""

            response = self.model.generate_content(prompt)
            priority = response.text.strip().upper()
            
            # Map Gemini's response to our categories
            priority_map = {
                'MUST': 'must',
                'SHOULD': 'should',
                'COULD': 'could',
                'WONT': 'wont',
                "WON'T": 'wont'
            }
            
            return priority_map.get(priority, 'should')  # Default to should if response is unexpected
        except Exception as e:
            logger.error(f"Error getting Gemini priority: {str(e)}")
            return self.auto_categorize_requirement(requirement)  # Fallback to rule-based categorization

    def get_gemini_batch_priority(self, requirements: list, batch_size: int = 3) -> dict:
        """Get priorities for a batch of requirements to reduce API calls"""
        try:
            # Split requirements into batches
            batches = [requirements[i:i + batch_size] for i in range(0, len(requirements), batch_size)]
            result = {}
            
            for batch in batches:
                try:
                    # Create a combined prompt for the batch
                    reqs_text = "\n\n".join([
                        f"Requirement {i+1}:\n"
                        f"Description: {req.get('requirement', '')}\n"
                        f"Type: {req.get('type', 'Not specified')}"
                        for i, req in enumerate(batch)
                    ])

                    prompt = f"""As a banking system requirements expert, analyze these requirements and categorize each using the MoSCoW method.
Consider:
1. Business Value: Critical to core banking operations?
2. Legal/Security Impact: Affects compliance/security?
3. Dependencies: Prerequisites for other features?
4. Implementation Complexity: Effort vs benefit?

Requirements to analyze:
{reqs_text}

Respond with ONLY requirement numbers and priorities, one per line, like this:
1: MUST
2: SHOULD
etc.
Use only these words: MUST, SHOULD, COULD, WONT"""

                    # Check cache first
                    cached_response = self.get_cached_response(reqs_text)
                    if cached_response:
                        print("Using cached response for batch...")
                        response_text = cached_response
                    else:
                        print("Getting new response from Gemini...")
                        response = self.model.generate_content(prompt)
                        response_text = response.text.strip()
                        # Cache the response
                        self.cache_response(reqs_text, response_text)

                    lines = response_text.split('\n')
                    
                    # Process each line of the response
                    for line in lines:
                        try:
                            req_num, priority = line.split(':')
                            req_num = int(req_num.strip()) - 1
                            if req_num < len(batch):
                                priority = priority.strip().upper()
                                # Map priority to our categories
                                priority_map = {
                                    'MUST': 'must',
                                    'SHOULD': 'should',
                                    'COULD': 'could',
                                    'WONT': 'wont',
                                    "WON'T": 'wont'
                                }
                                result[id(batch[req_num])] = priority_map.get(priority, 'should')
                        except (ValueError, IndexError):
                            continue
                    
                    # Add delay only if we made an API call
                    if not cached_response:
                        import time
                        time.sleep(2)
                    
                except Exception as e:
                    if "429" in str(e):  # Rate limit error
                        print("Rate limit reached, using rule-based categorization for remaining requirements...")
                        # Fall back to rule-based for all remaining requirements
                        for remaining_batch in batches[batches.index(batch):]:
                            for req in remaining_batch:
                                result[id(req)] = self.auto_categorize_requirement(req)
                        break  # Exit the loop since we'll use rule-based for the rest
                    else:
                        logger.error(f"Error processing batch: {str(e)}")
                        # Fall back to rule-based for this batch only
                        for req in batch:
                            result[id(req)] = self.auto_categorize_requirement(req)

            return result

        except Exception as e:
            logger.error(f"Error in batch prioritization: {str(e)}")
            return {id(req): self.auto_categorize_requirement(req) for req in requirements}

    def prioritize_requirements(self, all_requirements: dict) -> dict:
        """Prioritize requirements using MoSCoW method with batched Gemini analysis"""
        try:
            # Initialize prioritized requirements
            prioritized_reqs = {
                "must": [],
                "should": [],
                "could": [],
                "wont": []
            }

            # Collect all requirements
            requirements_list = []
            
            # Add functional requirements
            for req in all_requirements.get("requirements", {}).get("functional", []):
                requirements_list.append({
                    "requirement": req,
                    "type": "Functional",
                    "category": "Core Business Logic"
                })
            
            # Add non-functional requirements
            for req in all_requirements.get("requirements", {}).get("non_functional", []):
                requirements_list.append({
                    "requirement": req,
                    "type": "Non-Functional",
                    "category": "System Quality"
                })
            
            # Add selected innovative features
            for feature in all_requirements.get("innovative_features", {}).get("selected_features", []):
                requirements_list.append({
                    "requirement": feature["name"] + ": " + feature["description"],
                    "type": "Innovative Feature",
                    "category": "Enhancement"
                })

            print("\nStep 7: Requirements Prioritization (MoSCoW)")
            print("=========================================")
            print("\nAnalyzing requirements in batches using AI...\n")

            # Get priorities for all requirements in batches
            priorities = self.get_gemini_batch_priority(requirements_list)
            
            # Organize requirements by priority
            all_reqs = []
            for req in requirements_list:
                priority = priorities.get(id(req), self.auto_categorize_requirement(req))
                prioritized_reqs[priority].append(req)
                all_reqs.append((priority, req))

            # Display current categorization
            print("\nInitial AI-Based Prioritization:")
            for priority in ["must", "should", "could", "wont"]:
                print(f"\n{priority.upper()} HAVE:")
                for i, (_, req) in enumerate([r for r in all_reqs if r[0] == priority]):
                    req_index = len([r for r in all_reqs if r[0] == priority][:i])
                    print(f"[{req_index}] {req['requirement']}")

            # Allow user to modify priorities
            while True:
                print("\nWould you like to change any requirement priorities?")
                print("Enter requirement index and new priority (e.g., '5 M' for index 5 to MUST)")
                print("Or enter 'done' to finish")
                
                choice = input("\nYour choice: ").strip().lower()
                if choice == 'done':
                    break
                
                try:
                    idx, new_priority = choice.split()
                    idx = int(idx)
                    if idx < 0 or idx >= len(all_reqs):
                        print("Invalid index! Please try again.")
                        continue
                        
                    new_priority = new_priority.upper()
                    if new_priority not in ['M', 'S', 'C', 'W']:
                        print("Invalid priority! Use M, S, C, or W.")
                        continue
                        
                    # Get current priority and requirement
                    current_priority, req = all_reqs[idx]
                    
                    # Remove from current priority
                    prioritized_reqs[current_priority].remove(req)
                    
                    # Add to new priority
                    new_priority_map = {'M': 'must', 'S': 'should', 'C': 'could', 'W': 'wont'}
                    prioritized_reqs[new_priority_map[new_priority]].append(req)
                    
                    # Update the tracking list
                    all_reqs[idx] = (new_priority_map[new_priority], req)
                    
                    print(f"\nUpdated priority for: {req['requirement']}")
                    print(f"From: {current_priority.upper()} to: {new_priority_map[new_priority].upper()}")
                    
                except (ValueError, IndexError):
                    print("Invalid input format! Use 'index priority' (e.g., '5 M') or 'done'")

            return prioritized_reqs

        except Exception as e:
            logger.error(f"Error prioritizing requirements: {str(e)}")
            return {"must": [], "should": [], "could": [], "wont": []}

    def compile_requirements(self, explicit_reqs: dict, gap_responses: dict, clarifying_responses: dict) -> dict:
        """Step 5: Compile functional and non-functional requirements"""
        compilation_prompt = f"""Based on all gathered information, compile a comprehensive list of requirements.

        Explicit Requirements:
        {json.dumps(explicit_reqs, indent=2)}

        Gap Responses:
        {json.dumps(gap_responses, indent=2)}

        Clarifying Responses:
        {json.dumps(clarifying_responses, indent=2)}

        Organize requirements into:
        1. Functional Requirements (what the system must do)
        2. Non-Functional Requirements (how the system should perform)

        For each requirement:
        1. Provide a clear, testable statement
        2. Reference source (explicit, gap analysis, or clarification)
        3. Priority (High, Medium, Low)
        4. Dependencies if any

        Format as JSON:
        {{
            "functional": [
                {{
                    "id": "F1",
                    "statement": "Requirement statement",
                    "source": "Source of requirement",
                    "priority": "Priority level",
                    "dependencies": ["Other requirement IDs"],
                    "acceptance_criteria": ["Criteria 1", "Criteria 2"]
                }}
            ],
            "non_functional": [
                {{
                    "id": "NF1",
                    "category": "Category (Performance/Security/etc)",
                    "statement": "Requirement statement",
                    "metric": "Measurable metric",
                    "priority": "Priority level"
                }}
            ]
        }}
        """
        try:
            response = self.model.generate_content(compilation_prompt)
            return self.extract_json_from_response(response.text)
        except Exception as e:
            logger.error(f"Error compiling requirements: {str(e)}")
            return {"functional": [], "non_functional": []}

    def analyze_requirements(self, project_prompt: str) -> dict:
        """Main method to analyze requirements following the 7-step process"""
        if not project_prompt or not isinstance(project_prompt, str):
            raise ValueError("Project description is required and must be a string")

        if not self.model:
            raise ValueError("API not initialized. Please call setup_api first.")

        print("\n=== Software Requirements Specification Analysis ===\n")

        # Step 1: Extract Explicit Requirements
        print("\nStep 1: Extracting explicit requirements...")
        explicit_reqs = self.extract_explicit_requirements(project_prompt)
        if not explicit_reqs.get("categories"):
            print("Warning: No explicit requirements extracted")

        # Step 2: Perform Gap Analysis
        print("\nStep 2: Performing gap analysis...")
        gaps = self.perform_gap_analysis(project_prompt, explicit_reqs)
        if not gaps:
            print("Warning: No gaps identified")

        # Step 3: Generate and Ask Gap Questions
        print("\nStep 3: Addressing identified gaps...")
        gap_responses = {}
        if gaps:
            gap_questions = self.generate_gap_questions(gaps)
            for question in gap_questions:
                print(f"\nCategory: {question['category']}")
                print(f"Question: {question['question']}")
                print(f"Context: {question['context']}\n")
                print("Options:")
                for i, opt in enumerate(question["options"], 1):
                    print(f"{i}. {opt}")
                
                while True:
                    try:
                        choice = input("\nEnter option number: ")
                        if not choice:  # Handle empty input
                            print("Please enter a valid option number.")
                            continue
                            
                        choice = choice.strip()
                        if choice.isdigit() and 1 <= int(choice) <= len(question["options"]):
                            gap_responses[question["category"]] = {
                                "question": question["question"],
                                "answer": question["options"][int(choice) - 1]
                            }
                            break
                        print("Please enter a valid option number.")
                    except Exception as e:
                        print(f"Invalid input. Please try again: {str(e)}")
        else:
            print("No gaps to address")

        # Step 4: Generate and Ask Clarifying Questions
        print("\nStep 4: Asking clarifying questions...")
        clarifying_questions = self.generate_clarifying_questions(
            project_prompt, 
            explicit_reqs,
            gap_responses
        )
        
        clarifying_responses = {}
        if clarifying_questions:
            for aspect in clarifying_questions:
                print(f"\n=== {aspect['aspect']} ===")
                aspect_responses = []
                for q in aspect["questions"]:
                    print(f"\nQuestion: {q['question']}")
                    print(f"Context: {q['context']}")
                    print(f"Impact: {q['impact']}\n")
                    print("Options:")
                    for i, opt in enumerate(q["options"], 1):
                        print(f"{i}. {opt}")
                        
                    while True:
                        try:
                            opt_choice = input("\nEnter option number: ")
                            if not opt_choice:  # Handle empty input
                                print("Please enter a valid option number.")
                                continue
                                
                            opt_choice = opt_choice.strip()
                            if opt_choice.isdigit() and 1 <= int(opt_choice) <= len(q["options"]):
                                aspect_responses.append({
                                    "question": q["question"],
                                    "answer": q["options"][int(opt_choice) - 1]
                                })
                                break
                            print("Please enter a valid option number.")
                        except Exception as e:
                            print(f"Invalid input. Please try again: {str(e)}")
                clarifying_responses[aspect["aspect"]] = aspect_responses
        else:
            print("Warning: No clarifying questions generated")

        # Step 5: Compile Requirements
        print("\nStep 5: Compiling requirements...")
        compiled_requirements = self.compile_requirements(
            explicit_reqs,
            gap_responses,
            clarifying_responses
        )
        if not compiled_requirements.get("functional") and not compiled_requirements.get("non_functional"):
            print("Warning: No requirements compiled")

        # Step 6: Suggest Innovative Features
        print("\nStep 6: Suggesting innovative features...")
        innovative_features = self.suggest_innovative_features({
            "explicit_requirements": explicit_reqs,
            "gap_responses": gap_responses,
            "clarifying_responses": clarifying_responses,
            "compiled_requirements": compiled_requirements
        })
        
        if not innovative_features.get("selected_features"):
            print("Warning: No innovative features selected")
        else:
            print(f"Selected {len(innovative_features['selected_features'])} innovative features")

        # Step 7: Prioritize Requirements
        print("\nStep 7: Prioritizing requirements...")
        prioritized_requirements = self.prioritize_requirements({
            "requirements": compiled_requirements,
            "innovative_features": innovative_features
        })

        # Return all results for SRS formatting
        return {
            "explicit_requirements": explicit_reqs,
            "gaps": gaps,
            "gap_responses": gap_responses,
            "clarifying_responses": clarifying_responses,
            "requirements": compiled_requirements,
            "innovative_features": innovative_features,
            "prioritized_requirements": prioritized_requirements
        }

    def save_requirements(self, filename: str, analysis_results: dict):
        """Save the requirements following the 7-step structure"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== Software Requirements Specification ===\n\n")
            
            # 1. Explicit Requirements
            f.write("1. Explicit Requirements\n")
            f.write("=====================\n\n")
            for category in analysis_results.get("explicit_requirements", {}).get("categories", []):
                f.write(f"{category['name']}:\n")
                for req in category["requirements"]:
                    f.write(f"- [{req['type']}] {req['requirement']}\n")
                    f.write(f"  Source: {req['source']}\n\n")

            # 2. Gap Analysis
            f.write("\n2. Gap Analysis\n")
            f.write("=============\n\n")
            for gap in analysis_results.get("gaps", []):
                f.write(f"Category: {gap['category']}\n")
                f.write(f"Missing: {gap['gap']}\n")
                f.write(f"Importance: {gap['importance']}\n")
                f.write(f"Standard: {gap['standard']}\n")
                f.write(f"Risk: {gap['risk']}\n")
                f.write(f"Priority: {gap['priority']}\n\n")

            # 3. Gap Responses
            f.write("\n3. Gap Resolution\n")
            f.write("===============\n\n")
            for category, response in analysis_results.get("gap_responses", {}).items():
                f.write(f"Category: {category}\n")
                f.write(f"Question: {response['question']}\n")
                f.write(f"Decision: {response['answer']}\n\n")

            # 4. Clarifying Questions and Answers
            f.write("\n4. Requirement Clarifications\n")
            f.write("==========================\n\n")
            for q in analysis_results.get("clarifying_responses", {}).values():
                for qa in q:
                    f.write(f"Question: {qa['question']}\n")
                    f.write(f"Decision: {qa['answer']}\n\n")

            # 5. Known Requirements
            f.write("\n5. Known Requirements\n")
            f.write("===================\n")
            for req in analysis_results.get("requirements", {}).get("functional", []):
                f.write(f"- {req}\n")
            
            # 6. Requirement Gaps
            f.write("\n6. Requirement Gaps\n")
            f.write("=================\n")
            for gap in analysis_results.get("requirement_gaps", []):
                f.write(f"- Missing: {gap['gap']}\n")
                f.write(f"  Criticality: {gap['criticality']}\n")
                f.write(f"  Justification: {gap['justification']}\n")
                if gap.get("industry_standard"):
                    f.write(f"  Industry Standard: {gap['industry_standard']}\n")
                if gap.get("risk"):
                    f.write(f"  Risk: {gap['risk']}\n")
                f.write("\n")

            # 7. Detailed Requirements
            f.write("\n7. Detailed Requirements\n")
            f.write("======================\n")
            for category, qa_pairs in analysis_results.get("requirements", {}).items():
                if qa_pairs:
                    f.write(f"\n{category}:\n")
                    for qa in qa_pairs:
                        f.write(f"- {qa['statement']}\n")
            
            # 8. Selected Innovative Features
            f.write("\n8. Selected Innovative Features\n")
            f.write("============================\n")
            features = analysis_results.get("innovative_features", {}).get("selected_features", [])
            if features:
                for feature in features:
                    f.write(f"\nFeature: {feature['name']}\n")
                    f.write(f"Description: {feature['description']}\n")
                    f.write(f"Impact: {feature['impact']}\n")
                    f.write(f"Estimated Effort: {feature['estimated_effort']}\n")
                    
                    f.write("\nPrerequisites:\n")
                    for prereq in feature['prerequisites']:
                        f.write(f"- {prereq}\n")
                    
                    f.write("\nConfiguration Requirements:\n")
                    reqs = analysis_results.get("innovative_features", {}).get("feature_requirements", {}).get(feature['name'], [])
                    for req in reqs:
                        f.write(f"- {req['question']}: {req['selected_option']}\n")
                    f.write("\n")
            else:
                f.write("\nNo innovative features were selected.\n")

            # 9. Requirements Prioritization (MoSCoW)
            f.write("\n9. Requirements Prioritization (MoSCoW)\n")
            f.write("===================================\n")
            
            priorities = analysis_results.get("prioritized_requirements", {})
            if any(priorities.values()):
                # Calculate max lengths for table columns
                req_len = 60  # Fixed width for better formatting
                type_len = 15
                cat_len = 20

                # Header
                f.write("\n")
                f.write("+" + "-" * (req_len + 2) + "+" + "-" * (type_len + 2) + "+" + "-" * (cat_len + 2) + "+" + "-" * 10 + "+\n")
                f.write(f"| {'Requirement':<{req_len}} | {'Type':<{type_len}} | {'Category':<{cat_len}} | Priority |\n")
                f.write("+" + "=" * (req_len + 2) + "+" + "=" * (type_len + 2) + "+" + "=" * (cat_len + 2) + "+" + "=" * 10 + "+\n")

                # Must Have
                for req in priorities.get("must", []):
                    req_text = str(req.get('requirement', req.get('statement', '')))[:req_len]
                    req_type = str(req.get('type', ''))[:type_len]
                    req_category = str(req.get('category', ''))[:cat_len]
                    
                    f.write(f"| {req_text:<{req_len}} | {req_type:<{type_len}} | {req_category:<{cat_len}} | MUST     |\n")
                    f.write("+" + "-" * (req_len + 2) + "+" + "-" * (type_len + 2) + "+" + "-" * (cat_len + 2) + "+" + "-" * 10 + "+\n")

                # Should Have
                for req in priorities.get("should", []):
                    req_text = str(req.get('requirement', req.get('statement', '')))[:req_len]
                    req_type = str(req.get('type', ''))[:type_len]
                    req_category = str(req.get('category', ''))[:cat_len]
                    
                    f.write(f"| {req_text:<{req_len}} | {req_type:<{type_len}} | {req_category:<{cat_len}} | SHOULD   |\n")
                    f.write("+" + "-" * (req_len + 2) + "+" + "-" * (type_len + 2) + "+" + "-" * (cat_len + 2) + "+" + "-" * 10 + "+\n")

                # Could Have
                for req in priorities.get("could", []):
                    req_text = str(req.get('requirement', req.get('statement', '')))[:req_len]
                    req_type = str(req.get('type', ''))[:type_len]
                    req_category = str(req.get('category', ''))[:cat_len]
                    
                    f.write(f"| {req_text:<{req_len}} | {req_type:<{type_len}} | {req_category:<{cat_len}} | COULD    |\n")
                    f.write("+" + "-" * (req_len + 2) + "+" + "-" * (type_len + 2) + "+" + "-" * (cat_len + 2) + "+" + "-" * 10 + "+\n")

                # Won't Have
                for req in priorities.get("wont", []):
                    req_text = str(req.get('requirement', req.get('statement', '')))[:req_len]
                    req_type = str(req.get('type', ''))[:type_len]
                    req_category = str(req.get('category', ''))[:cat_len]
                    
                    f.write(f"| {req_text:<{req_len}} | {req_type:<{type_len}} | {req_category:<{cat_len}} | WON'T    |\n")
                    f.write("+" + "-" * (req_len + 2) + "+" + "-" * (type_len + 2) + "+" + "-" * (cat_len + 2) + "+" + "-" * 10 + "+\n")
            else:
                f.write("\nNo prioritized requirements available.\n")

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variable
    
    api_key = "AIzaSyCwFSFGaOV-XkZHji11nkm8Ptu29Y_-5io"
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        return

    analyzer = RequirementsAnalyzer()
    analyzer.setup_api(api_key)

    # Get project description
    print("\nWelcome to the Requirements Analysis Tool!")
    print("Please enter your project description below:")
    project_prompt = input("\nProject Description: ")
    
    try:
        # Analyze requirements and generate questions
        print("\nAnalyzing project requirements...")
        results = analyzer.analyze_requirements(project_prompt)
        
        # Save to file
        analyzer.save_requirements("requirements_answers.txt", results)
        print(f"\nRequirements have been saved to 'requirements_answers.txt'")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
