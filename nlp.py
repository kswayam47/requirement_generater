import os
import argparse
import re
import sys
import random
import logging
import json
import time
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from collections import Counter
from functools import lru_cache
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import spacy
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('requirements_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Data class to store analysis results."""
    domain: str
    system_type: str
    key_entities: List[str]
    key_actions: List[str]
    key_phrases: List[str]
    additional_insights: List[str]
    sentiment: float
    complexity: float
    technical_terms: List[str]
    business_terms: List[str]

class RateLimiter:
    """Rate limiter for API calls."""
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if we've exceeded the rate limit."""
        now = time.time()
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.calls.append(now)

class RequirementsGenerator:
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = ".cache"):
        # Load patterns and common terms only once
        self._load_patterns_and_terms()
        
        # Initialize cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize rate limiter
        self.rate_limiter = RateLimiter()

        # Initialize Gemini API if key is provided
        self.use_gemini = False
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro-1.5-latest')
                self.use_gemini = True
                logger.info("Gemini API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {str(e)}")
                self.use_gemini = False

        # Initialize spaCy for better NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def _load_patterns_and_terms(self):
        """Load regex patterns and common terms."""
        self.requirement_patterns = [
            r'should (\w+\s){1,5}',
            r'must (\w+\s){1,5}',
            r'needs to (\w+\s){1,5}',
            r'required to (\w+\s){1,5}',
            r'will (\w+\s){1,5}',
            r'shall (\w+\s){1,5}',
            r'functionality for (\w+\s){1,5}',
            r'ability to (\w+\s){1,5}',
            r'feature(s)? (\w+\s){1,5}',
            r'capability to (\w+\s){1,5}',
            r'provide(s)? (\w+\s){1,5}',
            r'enable(s)? (\w+\s){1,5}',
            r'support(s)? (\w+\s){1,5}',
            r'ensure(s)? (\w+\s){1,5}',
            r'allow(s)? (\w+\s){1,5}',
            r'permit(s)? (\w+\s){1,5}',
            r'require(s)? (\w+\s){1,5}',
            r'maintain(s)? (\w+\s){1,5}',
            r'validate(s)? (\w+\s){1,5}',
            r'verify(s)? (\w+\s){1,5}'
        ]
        
        # Compile regex patterns for better performance
        self.compiled_patterns = [re.compile(pattern) for pattern in self.requirement_patterns]
        
        self.common_entities = {
            'user', 'admin', 'system', 'application', 'data', 'interface',
            'patient', 'doctor', 'nurse', 'staff', 'hospital', 'clinic',
            'database', 'record', 'report', 'dashboard', 'module', 'function',
            'manager', 'administrator', 'client', 'server', 'platform', 'service',
            'tool', 'feature', 'capability', 'component', 'analyst', 'developer',
            'tester', 'auditor', 'operator', 'supervisor', 'coordinator',
            'customer', 'vendor', 'partner', 'stakeholder', 'end-user',
            'administrator', 'operator', 'analyst', 'developer', 'tester'
        }
        
        self.common_verbs = {
            'manage', 'create', 'view', 'edit', 'delete', 'update', 'track',
            'monitor', 'generate', 'book', 'schedule', 'register', 'login',
            'search', 'filter', 'sort', 'analyze', 'report', 'integrate',
            'export', 'import', 'upload', 'download', 'send', 'receive',
            'notify', 'alert', 'validate', 'approve', 'reject', 'process',
            'configure', 'customize', 'optimize', 'maintain', 'backup',
            'restore', 'archive', 'audit', 'review', 'verify', 'authenticate',
            'authorize', 'encrypt', 'decrypt', 'synchronize', 'replicate',
            'backup', 'restore', 'archive', 'compress', 'decompress'
        }
        
        # Domain identification lookup
        self.system_types = {'system', 'application', 'platform', 'tool', 'solution', 'software'}
        
        self.domains = {
            'healthcare': {'hospital', 'healthcare', 'medical', 'clinical', 'patient', 'doctor', 'health'},
            'finance': {'banking', 'financial', 'transaction', 'payment', 'account', 'investment', 'trading'},
            'education': {'education', 'learning', 'student', 'teacher', 'course', 'academic', 'university'},
            'retail': {'retail', 'store', 'shop', 'inventory', 'product', 'e-commerce', 'shopping'},
            'enterprise': {'enterprise', 'business', 'corporate', 'organization', 'company', 'workflow'},
            'technology': {'software', 'hardware', 'network', 'cloud', 'database', 'security'},
            'government': {'government', 'public', 'agency', 'department', 'service', 'citizen'},
            'manufacturing': {'manufacturing', 'production', 'factory', 'inventory', 'supply', 'quality'}
        }

    @lru_cache(maxsize=1000)
    def get_cached_analysis(self, text_hash: str) -> Optional[Dict]:
        """Get cached analysis results if available."""
        cache_file = self.cache_dir / f"{text_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cache: {str(e)}")
        return None

    def save_to_cache(self, text_hash: str, analysis: Dict):
        """Save analysis results to cache."""
        cache_file = self.cache_dir / f"{text_hash}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(analysis, f)
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")

    def analyze_with_gemini(self, text: str) -> Dict:
        """Analyze text using Gemini API for enhanced requirements extraction."""
        if not self.use_gemini:
            return {}

        try:
            # Check cache first
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cached_result = self.get_cached_analysis(text_hash)
            if cached_result:
                return cached_result

            # Rate limit API calls
            self.rate_limiter.wait_if_needed()

            prompt = f"""
            Perform a comprehensive software requirements analysis of the following text.
            Consider all aspects including technical, business, user needs, and industry standards.
            
            Text: {text}

            Please provide a detailed JSON response with the following structure:
            {{
                "domain": "main domain of the system",
                "system_type": "type of system",
                "key_entities": ["list of key entities/actors"],
                "key_actions": ["list of key actions/verbs"],
                "key_phrases": ["list of key requirement phrases"],
                "additional_insights": ["list of additional insights"],
                "sentiment": float between -1 and 1,
                "complexity": float between 0 and 1,
                "technical_terms": ["list of technical terms"],
                "business_terms": ["list of business terms"],
                "user_stories": [
                    {{
                        "role": "user role",
                        "action": "action to perform",
                        "benefit": "expected benefit",
                        "acceptance_criteria": ["list of acceptance criteria"]
                    }}
                ],
                "functional_requirements": [
                    {{
                        "id": "FR-001",
                        "description": "requirement description",
                        "priority": "high/medium/low",
                        "category": "category of requirement",
                        "dependencies": ["list of dependencies"],
                        "acceptance_criteria": ["list of acceptance criteria"],
                        "business_value": "description of business value",
                        "technical_impact": "description of technical impact"
                    }}
                ],
                "non_functional_requirements": [
                    {{
                        "id": "NFR-001",
                        "category": "performance/security/scalability/etc",
                        "description": "requirement description",
                        "priority": "high/medium/low",
                        "metrics": ["list of measurable metrics"],
                        "constraints": ["list of constraints"],
                        "validation_method": "method to validate requirement"
                    }}
                ],
                "technical_constraints": [
                    {{
                        "description": "constraint description",
                        "impact": "high/medium/low",
                        "mitigation": "possible mitigation strategies"
                    }}
                ],
                "business_constraints": [
                    {{
                        "description": "constraint description",
                        "impact": "high/medium/low",
                        "mitigation": "possible mitigation strategies"
                    }}
                ],
                "assumptions": [
                    {{
                        "description": "assumption description",
                        "impact": "high/medium/low",
                        "validation_method": "method to validate assumption"
                    }}
                ],
                "dependencies": [
                    {{
                        "description": "dependency description",
                        "type": "internal/external/third-party",
                        "criticality": "high/medium/low",
                        "mitigation": "dependency management strategy"
                    }}
                ],
                "risks": [
                    {{
                        "description": "risk description",
                        "impact": "high/medium/low",
                        "probability": "high/medium/low",
                        "mitigation": "mitigation strategy",
                        "contingency": "contingency plan"
                    }}
                ],
                "success_criteria": [
                    {{
                        "criterion": "success criterion description",
                        "measurement_method": "how to measure",
                        "target_value": "expected value",
                        "priority": "high/medium/low"
                    }}
                ],
                "timeline_considerations": [
                    {{
                        "phase": "project phase",
                        "activities": ["list of activities"],
                        "dependencies": ["list of dependencies"],
                        "estimated_duration": "duration estimate",
                        "critical_path": "is this on critical path?"
                    }}
                ],
                "resource_requirements": [
                    {{
                        "type": "resource type",
                        "description": "resource description",
                        "quantity": "required quantity",
                        "duration": "required duration",
                        "skills": ["required skills"],
                        "availability": "availability requirements"
                    }}
                ],
                "quality_requirements": [
                    {{
                        "category": "quality category",
                        "description": "requirement description",
                        "metrics": ["measurable metrics"],
                        "target_values": ["target values"],
                        "validation_method": "validation approach"
                    }}
                ],
                "security_requirements": [
                    {{
                        "category": "security category",
                        "description": "requirement description",
                        "compliance_standards": ["applicable standards"],
                        "implementation_guidelines": ["implementation guidelines"],
                        "validation_method": "validation approach"
                    }}
                ],
                "performance_requirements": [
                    {{
                        "category": "performance category",
                        "description": "requirement description",
                        "metrics": ["performance metrics"],
                        "target_values": ["target values"],
                        "conditions": ["test conditions"]
                    }}
                ],
                "integration_requirements": [
                    {{
                        "system": "system to integrate with",
                        "description": "integration description",
                        "interface_requirements": ["interface requirements"],
                        "data_requirements": ["data requirements"],
                        "security_requirements": ["security requirements"]
                    }}
                ],
                "maintenance_requirements": [
                    {{
                        "category": "maintenance category",
                        "description": "requirement description",
                        "frequency": "maintenance frequency",
                        "responsibilities": ["responsible parties"],
                        "procedures": ["maintenance procedures"]
                    }}
                ],
                "training_requirements": [
                    {{
                        "role": "user role",
                        "description": "training description",
                        "topics": ["training topics"],
                        "duration": "training duration",
                        "delivery_method": "training delivery method"
                    }}
                ],
                "documentation_requirements": [
                    {{
                        "type": "documentation type",
                        "description": "requirement description",
                        "format": "required format",
                        "audience": "target audience",
                        "delivery_method": "delivery method"
                    }}
                ],
                "compliance_requirements": [
                    {{
                        "standard": "compliance standard",
                        "description": "requirement description",
                        "requirements": ["specific requirements"],
                        "validation_method": "validation approach",
                        "documentation_needs": ["required documentation"]
                    }}
                ],
                "stakeholder_requirements": [
                    {{
                        "stakeholder": "stakeholder role",
                        "requirements": ["specific requirements"],
                        "expectations": ["stakeholder expectations"],
                        "success_criteria": ["success criteria"],
                        "communication_needs": ["communication requirements"]
                    }}
                ],
                "cost_considerations": [
                    {{
                        "category": "cost category",
                        "description": "cost description",
                        "estimation_method": "estimation method",
                        "factors": ["cost factors"],
                        "mitigation_strategies": ["cost mitigation strategies"]
                    }}
                ],
                "environmental_considerations": [
                    {{
                        "category": "environmental category",
                        "description": "requirement description",
                        "impact": "environmental impact",
                        "mitigation": "mitigation strategies",
                        "compliance_requirements": ["compliance requirements"]
                    }}
                ],
                "future_considerations": [
                    {{
                        "aspect": "future aspect",
                        "description": "consideration description",
                        "potential_impact": "potential impact",
                        "preparation_strategy": "preparation strategy",
                        "timeline": "expected timeline"
                    }}
                ]
            }}

            Consider the following aspects in your analysis:
            1. User needs and expectations
            2. Business objectives and value
            3. Technical feasibility and constraints
            4. Security and compliance requirements
            5. Performance and scalability needs
            6. Integration requirements
            7. Maintenance and support considerations
            8. Risk factors and mitigation strategies
            9. Timeline and resource implications
            10. Success criteria and acceptance criteria
            11. Quality assurance requirements
            12. Training and documentation needs
            13. Cost considerations
            14. Environmental impact
            15. Future scalability and extensibility
            16. Stakeholder management
            17. Change management
            18. Compliance and regulatory requirements
            19. Data management and privacy
            20. System architecture considerations
            21. Testing and validation requirements
            22. Deployment and operational considerations
            23. Disaster recovery and business continuity
            24. Vendor management and third-party dependencies
            25. Cultural and organizational impact
            26. Innovation and competitive advantage
            27. Sustainability and long-term viability
            28. Accessibility and inclusivity
            29. Internationalization and localization
            30. Ethical considerations and social impact

            Provide detailed, specific, and actionable information for each section.
            Consider industry best practices and standards relevant to the domain.
            Include specific metrics and measurable criteria where applicable.
            Address both immediate needs and future considerations.
            Consider the impact on all stakeholders and the broader ecosystem.
            """

            response = self.model.generate_content(prompt)
            if response.text:
                # Clean and parse the response
                cleaned_response = response.text.replace('```json', '').replace('```', '').strip()
                result = json.loads(cleaned_response)
                
                # Cache the result
                self.save_to_cache(text_hash, result)
                return result
            return {}
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {str(e)}")
            return {}

    def analyze_with_spacy(self, text: str) -> Dict:
        """Analyze text using spaCy for additional insights."""
        doc = self.nlp(text)
        
        # Extract named entities
        entities = [ent.text for ent in doc.ents]
        
        # Extract key phrases (noun phrases)
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Analyze sentiment (simple implementation)
        sentiment = sum([token.sentiment for token in doc]) / len(doc)
        
        # Calculate complexity based on sentence length and word complexity
        complexity = len(doc) / 100  # Normalize to 0-1 range
        
        return {
            "entities": entities,
            "key_phrases": key_phrases,
            "sentiment": sentiment,
            "complexity": complexity
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize the input text."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract potential key phrases from text using compiled regex."""
        if not text:
            return []
            
        text_lower = text.lower()
        key_phrases = set()
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text_lower)
            key_phrases.update(matches)
        
        return list(key_phrases)

    def extract_entities(self, text: str) -> List[str]:
        """Extract likely entities (nouns) that could be system actors or components."""
        if not text:
            return []
            
        text_lower = text.lower()
        entity_counts = Counter()
        
        words = set(text_lower.split())
        for entity in self.common_entities:
            if entity in text_lower:
                count = text_lower.count(entity)
                entity_counts[entity] = count
        
        # Return sorted entities by frequency
        return [entity for entity, _ in entity_counts.most_common()]

    def extract_verbs(self, text: str) -> List[str]:
        """Extract action verbs that might indicate system capabilities."""
        if not text:
            return []
            
        text_lower = text.lower()
        verb_counts = Counter()
        
        words = set(text_lower.split())
        for verb in self.common_verbs:
            if verb in text_lower:
                count = text_lower.count(verb)
                verb_counts[verb] = count
        
        # Return sorted verbs by frequency
        return [verb for verb, _ in verb_counts.most_common()]

    def identify_domain(self, text: str) -> Tuple[str, str]:
        """Identify the main system type and domain."""
        text_lower = text.lower()
        
        # Find system type
        main_system = next((e for e in self.system_types if e in text_lower), 'system')
        
        # Find domain
        for domain, keywords in self.domains.items():
            if any(keyword in text_lower for keyword in keywords):
                return main_system, domain
        
        return main_system, 'general'

    def generate_requirements_statement(self, text: str, format_type: str = 'standard') -> str:
        """Generate a formatted requirements statement based on text analysis."""
        try:
            cleaned_text = self.clean_text(text)
            
            # Get Gemini analysis if available
            gemini_analysis = {}
            if self.use_gemini:
                gemini_analysis = self.analyze_with_gemini(cleaned_text)
                logger.debug(f"Gemini analysis: {gemini_analysis}")
            
            # Get spaCy analysis
            spacy_analysis = self.analyze_with_spacy(cleaned_text)
            
            # Extract various components from the text
            entities = self.extract_entities(cleaned_text)
            verbs = self.extract_verbs(cleaned_text)
            key_phrases = self.extract_key_phrases(cleaned_text)
            
            # Use Gemini analysis if available, otherwise use traditional methods
            if gemini_analysis:
                main_system = gemini_analysis.get('system_type', 'system')
                domain = gemini_analysis.get('domain', 'general')
                entities = gemini_analysis.get('key_entities', entities)
                verbs = gemini_analysis.get('key_actions', verbs)
                key_phrases = gemini_analysis.get('key_phrases', key_phrases)
            else:
                main_system, domain = self.identify_domain(cleaned_text)
            
            # Format the primary entities (actors/users)
            primary_entities = [e for e in entities[:4] if e not in ['system', 'application', 'platform']]
            
            # Format the primary actions/capabilities
            primary_actions = verbs[:6]
            
            if format_type == 'detailed':
                # Generate a comprehensive requirements document
                requirements = []
                
                # System Overview
                requirements.append("1. SYSTEM OVERVIEW")
                requirements.append("==================")
                requirements.append(f"The {domain} {main_system} is designed to serve the following purposes:")
                
                # Key Capabilities
                requirements.append("\n2. KEY CAPABILITIES")
                requirements.append("==================")
                if primary_entities and primary_actions:
                    for entity, action in zip(primary_entities[:3], primary_actions[:3]):
                        requirements.append(f"- {entity.capitalize()}s can {action}")
                
                # User Stories
                if gemini_analysis and 'user_stories' in gemini_analysis:
                    requirements.append("\n3. USER STORIES")
                    requirements.append("===============")
                    for story in gemini_analysis['user_stories'][:3]:
                        requirements.append(f"As a {story['role']}")
                        requirements.append(f"I want to {story['action']}")
                        requirements.append(f"So that {story['benefit']}\n")
                
                # Functional Requirements
                if gemini_analysis and 'functional_requirements' in gemini_analysis:
                    requirements.append("\n4. FUNCTIONAL REQUIREMENTS")
                    requirements.append("=========================")
                    for req in gemini_analysis['functional_requirements'][:5]:
                        requirements.append(f"{req['id']} [{req['priority']}] - {req['description']}")
                        requirements.append(f"Category: {req['category']}\n")
                
                # Non-Functional Requirements
                if gemini_analysis and 'non_functional_requirements' in gemini_analysis:
                    requirements.append("\n5. NON-FUNCTIONAL REQUIREMENTS")
                    requirements.append("===============================")
                    for req in gemini_analysis['non_functional_requirements'][:5]:
                        requirements.append(f"{req['id']} [{req['category']}] - {req['description']}")
                        requirements.append(f"Priority: {req['priority']}\n")
                
                # Technical Analysis
                requirements.append("\n6. TECHNICAL ANALYSIS")
                requirements.append("=====================")
                requirements.append(f"- Sentiment Analysis: {spacy_analysis['sentiment']:.2f}")
                requirements.append(f"- Complexity Score: {spacy_analysis['complexity']:.2f}")
                
                if gemini_analysis:
                    if 'technical_terms' in gemini_analysis:
                        requirements.append("\nTechnical Terms:")
                        for term in gemini_analysis['technical_terms'][:5]:
                            requirements.append(f"- {term}")
                    
                    if 'technical_constraints' in gemini_analysis:
                        requirements.append("\nTechnical Constraints:")
                        for constraint in gemini_analysis['technical_constraints'][:3]:
                            requirements.append(f"- {constraint}")
                
                # Business Analysis
                if gemini_analysis:
                    requirements.append("\n7. BUSINESS ANALYSIS")
                    requirements.append("===================")
                    if 'business_terms' in gemini_analysis:
                        requirements.append("\nBusiness Terms:")
                        for term in gemini_analysis['business_terms'][:5]:
                            requirements.append(f"- {term}")
                    
                    if 'business_constraints' in gemini_analysis:
                        requirements.append("\nBusiness Constraints:")
                        for constraint in gemini_analysis['business_constraints'][:3]:
                            requirements.append(f"- {constraint}")
                
                # Project Considerations
                if gemini_analysis:
                    requirements.append("\n8. PROJECT CONSIDERATIONS")
                    requirements.append("=========================")
                    
                    if 'assumptions' in gemini_analysis:
                        requirements.append("\nAssumptions:")
                        for assumption in gemini_analysis['assumptions'][:3]:
                            requirements.append(f"- {assumption}")
                    
                    if 'dependencies' in gemini_analysis:
                        requirements.append("\nDependencies:")
                        for dependency in gemini_analysis['dependencies'][:3]:
                            requirements.append(f"- {dependency}")
                    
                    if 'risks' in gemini_analysis:
                        requirements.append("\nRisks:")
                        for risk in gemini_analysis['risks'][:3]:
                            requirements.append(f"- {risk['description']}")
                            requirements.append(f"  Impact: {risk['impact']}")
                            requirements.append(f"  Mitigation: {risk['mitigation']}")
                    
                    if 'success_criteria' in gemini_analysis:
                        requirements.append("\nSuccess Criteria:")
                        for criterion in gemini_analysis['success_criteria'][:3]:
                            requirements.append(f"- {criterion}")
                    
                    if 'timeline_considerations' in gemini_analysis:
                        requirements.append("\nTimeline Considerations:")
                        for consideration in gemini_analysis['timeline_considerations'][:3]:
                            requirements.append(f"- {consideration}")
                    
                    if 'resource_requirements' in gemini_analysis:
                        requirements.append("\nResource Requirements:")
                        for resource in gemini_analysis['resource_requirements'][:3]:
                            requirements.append(f"- {resource}")
                
                return "\n".join(requirements)
            
            else:
                # Generate a concise 5-6 line output with detailed SRS information
                lines = []
                
                # Line 1: System summary with core purpose and key actors
                actors = ", ".join([e for e in primary_entities[:3]])
                core_purpose = primary_actions[0] if primary_actions else "perform operations"
                
                # Ensure we don't truncate the summary in a confusing way
                summary = f"A {domain} {main_system} for {actors} to {core_purpose}"
                if cleaned_text and len(cleaned_text) > 20:  # Only add if there's substantial text
                    summary += f" that will address the requirements specified in the input text."
                lines.append(summary)
                
                # Line 2: Functional capabilities and main features
                if len(primary_actions) > 1:
                    capabilities = ", ".join(primary_actions[1:4])
                    lines.append(f"The system will provide capabilities for {capabilities} while ensuring compliance with industry standards and data security.")
                
                # Line 3: Technical specifications and constraints
                tech_specs = []
                if gemini_analysis and gemini_analysis.get('technical_constraints'):
                    for constraint in gemini_analysis['technical_constraints'][:2]:
                        tech_specs.append(f"{constraint['description']}")
                if not tech_specs and gemini_analysis and gemini_analysis.get('technical_terms'):
                    tech_specs = gemini_analysis['technical_terms'][:2]
                if tech_specs:
                    lines.append(f"Technical considerations include {'; '.join(tech_specs)}.")
                else:
                    lines.append(f"The implementation must ensure data security, robust performance, and scalability.")
                
                # Line 4: Integration and compliance requirements
                integrations = []
                if gemini_analysis and gemini_analysis.get('integration_requirements'):
                    for integration in gemini_analysis['integration_requirements'][:2]:
                        integrations.append(f"{integration['system']}")
                compliance = []
                if gemini_analysis and gemini_analysis.get('compliance_requirements'):
                    for comp in gemini_analysis['compliance_requirements'][:2]:
                        compliance.append(comp['standard'])
                
                if integrations or compliance:
                    int_text = f"The system will integrate with {', '.join(integrations)}" if integrations else "The system will support external integrations"
                    comp_text = f" while complying with {', '.join(compliance)}" if compliance else " and follow relevant regulatory requirements"
                    lines.append(f"{int_text}{comp_text}.")
                else:
                    lines.append("The system requires integration capabilities and compliance with industry-specific regulations.")
                
                # Line 5: User experience and performance requirements
                perf_reqs = []
                if gemini_analysis and gemini_analysis.get('performance_requirements'):
                    for req in gemini_analysis['performance_requirements'][:2]:
                        perf_reqs.append(f"{req['description']}")
                if perf_reqs:
                    lines.append(f"Performance requirements include {'; '.join(perf_reqs)}.")
                else:
                    lines.append("The system should maintain optimal performance under high user concurrency and data volume conditions.")
                
                # Line 6: Future considerations and scalability
                future = []
                if gemini_analysis and gemini_analysis.get('future_considerations'):
                    for f in gemini_analysis['future_considerations'][:2]:
                        future.append(f"{f['description']}")
                if future:
                    lines.append(f"Future enhancements should include {'; '.join(future)}.")
                else:
                    lines.append("The design should accommodate future scalability, feature extensions, and evolving business needs.")
                
                return "\n".join(lines)
                
        except Exception as e:
            logger.error(f"Error generating requirements statement: {str(e)}")
            raise

def read_input_file(file_path: str) -> str:
    """Read input from a file with proper error handling."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate formatted requirements statements from text')
    parser.add_argument('--file', '-f', help='Path to text file containing requirements')
    parser.add_argument('--text', '-t', help='Input text to process')
    parser.add_argument('--format', '-fmt', choices=['standard', 'detailed'], 
                       default='standard', help='Output format style')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--api-key', help='Google Gemini API key')
    parser.add_argument('--cache-dir', default='.cache', help='Directory for caching results')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--single-paragraph', '-sp', action='store_true', help='Display output as a single paragraph')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Get input text
        if args.file:
            input_text = read_input_file(args.file)
        elif args.text:
            input_text = args.text
        else:
            print("Enter your text (Ctrl+D to finish):")
            input_text = sys.stdin.read()
        
        if not input_text.strip():
            raise ValueError("No input text provided")
        
        # Process the text
        generator = RequirementsGenerator(api_key=args.api_key, cache_dir=args.cache_dir)
        requirements = generator.generate_requirements_statement(input_text, args.format)
        
        # Output the results
        print("\nGenerated Requirements Statement:")
        print("==================================")
        
        # Format the output for standard format
        if args.format == 'standard':
            # Convert multiline to a cohesive paragraph, preserving meaning
            lines = [line.strip() for line in requirements.split('\n') if line.strip()]
            
            # Apply quick text transformations using a dictionary of replacements
            replacements = {
                r'^The system (will|should|requires|must)': r'It \1', 
                r' the system ': r' it ',
                r'\.\.': r'.',
                r' \. ': r'. ',
                r' \s+': r' '
            }
            
            # Process each line
            for i in range(1, len(lines)):
                # Add periods if missing
                if not lines[i-1].endswith(('.', '!', '?')):
                    lines[i-1] += '.'
                
                # Avoid redundant "the system" phrases
                if i > 1 and lines[i].startswith("The system") and "system" in lines[i-1].lower():
                    for pattern, replacement in replacements.items():
                        lines[i] = re.sub(pattern, replacement, lines[i])
            
            # Join into paragraph
            paragraph = " ".join(lines)
            
            # Clean up final text
            paragraph = re.sub(r' +', ' ', paragraph)
            paragraph = re.sub(r' \.', '.', paragraph)
            paragraph = re.sub(r'\.\s+\.', '.', paragraph)
            paragraph = re.sub(r'[\.\s]+$', '.', paragraph)
            
            print(paragraph)
        else:
            # For detailed format, maintain the original formatting
            print(requirements)
            
        print("==================================")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
