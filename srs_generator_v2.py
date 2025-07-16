import os
from datetime import datetime
import google.generativeai as genai

class SRSGeneratorV2:
    def __init__(self):
        self.model = None
        self.setup_api()
        self.template = """# Software Requirements Specification
For {system_name}
Version 1.0

{date}

{content}"""

    def setup_api(self):
        """Setup the Gemini API with optimal parameters for document generation"""
        try:
            api_key = "AIzaSyBLFt5W46-q7y5iWFePiJ9CGLrx_W0hFaI"
            genai.configure(api_key=api_key)
            
            generation_config = {
                "temperature": 0.1,  # Very low temperature for precise output
                "top_p": 0.7,
                "top_k": 20,
                "max_output_tokens": 2048,  # Limited to ensure conciseness
            }
            
            self.model = genai.GenerativeModel('gemini-2.0-flash', generation_config=generation_config)
            print("Successfully configured Gemini 2.0 Flash API")
            
        except Exception as e:
            print(f"Error setting up API: {str(e)}")
            raise

    def parse_requirements(self, content):
        """Parse requirements from text format"""
        sections = {}
        current_section = None
        current_subsection = None
        requirements = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('==='):
                continue
                
            if line.endswith('==='):
                continue
                
            if line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
                current_section = line.split('.', 1)[1].strip()
                sections[current_section] = []
                continue
                
            if line.endswith(':'):
                current_subsection = line[:-1].strip()
                continue
                
            if line.startswith('-'):
                req = line[1:].strip()
                if current_subsection:
                    requirements.append({
                        'category': current_subsection,
                        'requirement': req
                    })
                    
            if line.startswith('Question:'):
                current_subsection = 'Questions'
                continue
                
            if line.startswith('Decision:'):
                if current_subsection == 'Questions':
                    requirements.append({
                        'category': 'Decisions',
                        'requirement': line.replace('Decision:', '').strip()
                    })
                    
        return requirements

    def extract_system_name(self, requirements):
        """Extract system name from requirements"""
        try:
            # First look for banking-specific keywords in requirements
            banking_keywords = ['banking', 'bank', 'financial institution', 'finance']
            
            # Check categories first
            for req in requirements[:10]:
                if isinstance(req, dict) and 'category' in req:
                    category = req['category'].lower()
                    # Check for banking terms
                    for keyword in banking_keywords:
                        if keyword in category:
                            return "Banking System"
                    
            # Then check requirements text
            for req in requirements:
                if isinstance(req, dict) and 'requirement' in req:
                    text = req['requirement'].lower()
                    # Check for banking-specific content
                    if any(keyword in text for keyword in banking_keywords):
                        return "Banking System"
                    # Look for account/transaction related terms
                    if ('account' in text and 'transactions' in text) or \
                       ('financial' in text and 'transactions' in text):
                        return "Banking System"
            
            # If no banking system detected, fall back to general detection
            for req in requirements[:10]:
                if isinstance(req, dict) and 'category' in req:
                    # Common domain indicators
                    domain_indicators = ['system', 'platform', 'application', 'service']
                    category = req['category'].lower()
                    for indicator in domain_indicators:
                        if indicator in category:
                            return req['category'].replace(':', '').strip()
                            
            # Look in requirements text for system type
            for req in requirements:
                if isinstance(req, dict) and 'requirement' in req:
                    text = req['requirement'].lower()
                    if 'system' in text or 'platform' in text or 'application' in text:
                        words = req['requirement'].split()
                        for i, word in enumerate(words):
                            if word.lower() in ['system', 'platform', 'application']:
                                start = max(0, i-2)
                                end = min(len(words), i+3)
                                return ' '.join(words[start:end])
            
            # Default if no system name found
            return "Software System"
            
        except Exception as e:
            print(f"Error extracting system name: {str(e)}")
            return "Software System"

    def generate_srs(self, input_file: str, output_file: str):
        """Generate a concise SRS document from requirements"""
        try:
            # Read requirements
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse requirements
            requirements = self.parse_requirements(content)
            
            # Extract system name
            system_name = self.extract_system_name(requirements)

            # Create focused prompt for document generation
            prompt = f"""Generate a precise and concise SRS document for the {system_name} using ONLY the following 10 sections. Focus on essential information and avoid any repetition.

Requirements:
{requirements}

CRITICAL RULES:
1. Generate exactly 2-3 pages in markdown
2. Use precise, professional language
3. NO duplication of requirements
4. Each requirement must be atomic and clear
5. Use consistent terminology

REQUIRED SECTIONS (EXACTLY IN THIS ORDER):
1. Purpose Section
   - Clear statement of system's main goal
   - Business value and objectives

2. Scope Section
   - System boundaries
   - Included features
   - Excluded features

3. Stakeholders Section
   - All identified stakeholders
   - Their roles and responsibilities

4. Features Section
   - Core system features
   - Key capabilities

5. Functional Requirements Section
   - All functional requirements
   - Grouped by category
   - Format: [FR-ID]: [Statement] [Priority]

6. Non-Functional Requirements Section
   - All non-functional requirements
   - Performance, reliability, etc.
   - Format: [NFR-ID]: [Statement] [Priority]

7. Security Requirements Section
   - All security-related requirements
   - Format: [SR-ID]: [Statement] [Priority]

8. Constraints Section
   - Technical limitations
   - Business rules
   - Regulatory requirements

9. Priorities Section (MoSCoW)
   - Must Have
   - Should Have
   - Could Have
   - Won't Have

10. Additional Section
    - Any requirements not covered above
    - Future considerations
    - DO NOT add implementation details

FORMAT RULES:
1. Use ## for main sections (e.g., ## 1. Purpose)
2. Use bullet points for lists
3. Requirements format: [ID]: [Statement] [Priority]
4. Maximum 2-3 sentences per item
5. Include source reference for traceability

CONSTRAINTS:
1. NO implementation details
2. NO repetition of requirements
3. NO placeholder text
4. MUST follow the exact section order given above"""

            # Generate SRS content
            response = self.model.generate_content(prompt)
            if not response.text:
                raise Exception("Empty response received from the model")

            # Format the document
            formatted_srs = self.template.format(
                system_name=system_name,
                date=datetime.now().strftime("%B %d, %Y"),
                content=response.text.strip()
            )

            # Validate content length (rough estimate: 2-3 pages ≈ 1000-1500 words)
            word_count = len(formatted_srs.split())
            if word_count > 1500:
                print(f"Warning: Document length ({word_count} words) exceeds target range")

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_srs)
            
            print(f"Successfully generated SRS document: {output_file}")
            print(f"Document length: {word_count} words")
            return True

        except Exception as e:
            print(f"Error generating SRS: {str(e)}")
            return False

def main():
    generator = SRSGeneratorV2()
    generator.generate_srs('requirements_answers.txt', 'system_srs.md')

if __name__ == "__main__":
    main()
