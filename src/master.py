"""
Monte Carlo Tree Search based solution generation framework.
Core implementation focusing on essential functionality.
"""

import math
import random
import time
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import networkx as nx
from config import config

class Agent:
    """
    Individual agent that performs solution generation and evaluation.
    """
    def __init__(
        self,
        agent_id: int,
        parent: Optional['Agent'],
        question: str,
        context: List[Document] = None,
        llm: ChatOpenAI = None
    ):
        self.agent_id = agent_id
        self.parent = parent
        self.question = question
        self.context = context or []
        self.llm = llm
        
        self.solution = ""
        self.validation = ""
        self.assessment = ""
        self.score = 0.0
        self.confidence = 0.0
        self.children = []
        self.passed = False
        self.terminal = False
        
        # MCTS statistics
        self.visit_count = 1
        self.avg_reward = 0.0
        
        # Track creation time for time-based features
        self._creation_time = time.time()
    
    def generate_solution(self):
        """Generate solution using context and question."""
        context_text = "\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(self.context)
        ]) if self.context else ""
        
        prompt = f"""Context:
{context_text}

Question: {self.question}

Generate a detailed step-by-step solution:"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)
        self.solution = response.content
    
    def validate(self):
        """Validate solution for correctness."""
        prompt = f"""Solution: {self.solution}

Validate the solution by checking:
1. Logical consistency
2. Completeness of answer
3. Clarity of explanation

Provide a validation analysis:"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)
        self.validation = response.content
    
    def assess(self):
        """Rate solution quality."""
        prompt = f"""Solution: {self.solution}
Validation: {self.validation}

Rate this solution:
1. Correctness (0-10): How accurate and valid
2. Confidence (0-10): How sure are we of this assessment

Provide ratings in this format:
Score: <number>
Confidence: <number>"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)
        
        try:
            text = response.content.lower()
            score_line = [line for line in text.split('\n') if 'score:' in line][0]
            conf_line = [line for line in text.split('\n') if 'confidence:' in line][0]
            
            self.score = float(score_line.split('score:')[1].strip())
            self.confidence = float(conf_line.split('confidence:')[1].strip()) / 10.0
            
        except Exception:
            self.score = random.uniform(0, 10)
            self.confidence = random.uniform(0, 1)
    
    def evaluate(self):
        """Determine if solution is complete and correct using weighted multi-criteria evaluation."""
        # Weighted criteria evaluation
        criteria_weights = {
            'final_marker': 0.2,   # Explicit marker is least important
            'completeness': 0.4,   # Completeness is crucial
            'correctness': 0.4     # Correctness equally important
        }
        
        criteria_scores = {
            'final_marker': float(self._check_final_marker()),
            'completeness': float(self._check_completeness()),
            'correctness': float(self._check_correctness())
        }
        
        # Calculate weighted score
        weighted_score = sum(
            score * criteria_weights[criterion]
            for criterion, score in criteria_scores.items()
        )
        
        # Time-based confidence degradation
        if hasattr(self, '_creation_time'):
            time_penalty = (time.time() - self._creation_time) / 3600  # Hours
            self.confidence = max(0.1, self.confidence * math.exp(-0.1 * time_penalty))
        
        # Advanced quality thresholds
        quality_threshold = 0.7  # Base threshold
        confidence_boost = self.confidence * 0.3  # Confidence can boost threshold
        
        self.passed = (
            weighted_score >= quality_threshold and
            self.score >= 7.0 and
            (self.confidence >= 0.7 or
             (self.score >= 8.5 and self.confidence >= 0.6))  # High score can compensate
        )
        
        # Store evaluation metrics for debugging
        self.evaluation_metrics = {
            'weighted_score': weighted_score,
            'criteria_scores': criteria_scores,
            'confidence': self.confidence,
            'quality_threshold': quality_threshold
        }

    def _check_final_marker(self) -> bool:
        """Check if solution has explicit final answer marker."""
        return "final answer:" in self.solution.lower()
    
    def _check_completeness(self) -> bool:
        """Verify solution addresses all aspects of the question."""
        prompt = f"""Question: {self.question}
Solution: {self.solution}

Does this solution address ALL aspects of the question? Consider:
1. All parts of the question are answered
2. No missing information
3. Solution is detailed enough

Respond with only True or False."""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)
        return "true" in response.content.lower()
    
    def _check_correctness(self) -> bool:
        """Verify solution is logically sound and accurate."""
        prompt = f"""Question: {self.question}
Solution: {self.solution}
Validation: {self.validation}

Is this solution logically sound and accurate? Consider:
1. No contradictions or errors
2. Reasoning is valid
3. Conclusions are supported

Respond with only True or False."""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)
        return "true" in response.content.lower()

class MASTER:
    """MASTER Framework: Monte Carlo Tree Search for solution generation."""
    
    def __init__(self, question: str,
                 context: List[Document] = None,
                 max_expansion: int = 3,
                 num_branches: int = 2,
                 llm: ChatOpenAI = None,
                 knowledge_graph: nx.Graph = None):
        """
        Args:
            question: The main user query
            context: Optional list of relevant documents
            max_expansion: Number of expansion rounds
            num_branches: Number of child agents per expansion
            llm: Language model to use
            knowledge_graph: NetworkX graph for knowledge base
        """
        self.llm = llm or ChatOpenAI(model=config.OPENAI.chat_model,
                                   temperature=config.OPENAI.temperature)
        self.knowledge_graph = knowledge_graph
        
        # Extract initial entities and enhance context
        initial_entities = self._extract_entities(question)
        enhanced_context = self._enhance_initial_context(context, initial_entities)
        
        self.root = Agent(0, None, question, enhanced_context, self.llm)
        self.agents = {0: self.root}
        self.max_expansion = max_expansion
        self.num_branches = num_branches
        self.current_id = 1
        
    def _enhance_initial_context(self, context: List[Document], entities: List[str]) -> List[Document]:
        """Enhance initial context with knowledge graph insights."""
        if not context:
            context = []
            
        if self.knowledge_graph and entities:
            # Get subgraph insights
            kg_insights = []
            for entity in entities:
                subgraph_content = self._get_entity_subgraph(entity)
                if subgraph_content:
                    kg_insights.append(subgraph_content)
                    
            # Add knowledge graph context
            if kg_insights:
                context.append(Document(
                    page_content="Initial Knowledge Graph Context:\n" + "\n".join(kg_insights)
                ))
                
            # Add context template if available
            template = self._get_context_template(entities)
            if template:
                context.append(Document(
                    page_content="Solution Framework:\n" + template
                ))
                
        return context
    
    def uct(self, agent: Agent) -> float:
        """Calculate UCT score with adaptive exploration and semantic context."""
        if agent.parent is None:
            return float('-inf')
            
        # Enhanced exploitation with recency bias
        if hasattr(agent, '_creation_time'):
            hours_old = (time.time() - agent._creation_time) / 3600
            recency_factor = math.exp(-0.05 * hours_old)  # Gradual time decay
        else:
            recency_factor = 1.0
            
        # Quality-weighted exploitation using multiple factors
        exploitation = (
            agent.confidence * agent.score * recency_factor +
            (1 - agent.confidence) * agent.avg_reward
        ) / 10.0  # Normalize to [0,1]
        
        # Progressive widening for better tree growth
        progressive_width = math.log(1 + agent.parent.visit_count) / (1 + agent.visit_count)
        
        # Depth-aware exploration
        depth = self._get_agent_depth(agent)
        depth_factor = 1.0 / (1 + depth)  # Reduce exploration at deeper levels
        
        # Performance-based exploration boost
        if agent.visit_count > 1:
            # Reward improvement over parent
            improvement = max(0, agent.avg_reward - agent.parent.avg_reward)
            performance_factor = math.sqrt(improvement + 0.1)
        else:
            performance_factor = 1.0
            
        # Enhanced UCT exploration term
        exploration = (
            math.sqrt(2) *
            performance_factor *
            depth_factor *
            math.sqrt(progressive_width)
        )
        
        # Dynamic exploration-exploitation balance
        visit_ratio = agent.visit_count / max(1, agent.parent.visit_count)
        exploration_weight = 1.0 / (1 + visit_ratio)  # Smooth decrease in exploration
        
        return exploitation + exploration_weight * exploration
        
    def _get_agent_depth(self, agent: Agent) -> int:
        """Calculate agent's depth in the tree."""
        depth = 0
        current = agent
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    def select_agent(self) -> Agent:
        """Select agent with highest UCT for expansion."""
        candidates = [a for a in self.agents.values() if not a.terminal]
        if not candidates:
            return self.root
        return max(candidates, key=lambda a: self.uct(a))
    
    def expand(self, agent: Agent):
        """Create child agents with dynamic branching and retry mechanism."""
        # Calculate number of branches based on agent performance
        dynamic_branches = self._calculate_branches(agent)
        max_retries = 2  # Maximum number of retries for failed expansions
        
        for i in range(dynamic_branches):
            child_agent = None
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    # Create child with filtered context
                    child_agent = Agent(
                        self.current_id + i,
                        agent,
                        agent.question,
                        self._filter_context(agent),
                        self.llm
                    )
                    
                    # Generate and evaluate solution
                    child_agent.generate_solution()
                    child_agent.validate()
                    child_agent.assess()
                    
                    # If solution quality is good, break retry loop
                    if child_agent.score >= 5.0 and child_agent.confidence >= 0.5:
                        break
                        
                    retry_count += 1
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        # Create a failed agent after all retries exhausted
                        child_agent = Agent(
                            self.current_id + i,
                            agent,
                            agent.question,
                            agent.context,
                            self.llm
                        )
                        child_agent.solution = f"Error after {max_retries} retries: {str(e)}"
                        child_agent.score = 0.0
                        child_agent.confidence = 0.0
                        child_agent.terminal = True
                        child_agent.passed = False
                        break
            
            if child_agent:
                # Check for terminal state
                if child_agent.score >= 8.0 and child_agent.confidence >= 0.8:
                    child_agent.terminal = True
                    child_agent.evaluate()
                
                self.agents[child_agent.agent_id] = child_agent
                agent.children.append(child_agent)
                
                if child_agent.terminal and not child_agent.passed:
                    self.backpropagate(child_agent)
        
        self.current_id += dynamic_branches

    def _calculate_branches(self, agent: Agent) -> int:
        """Calculate number of branches based on agent performance."""
        if agent.score >= 8.0 and agent.confidence >= 0.8:
            # High performing agents get fewer branches to focus exploitation
            return max(1, self.num_branches - 1)
        elif agent.score <= 3.0 or agent.confidence <= 0.3:
            # Low performing agents get more branches to encourage exploration
            return min(self.num_branches + 1, 4)
        return self.num_branches

    def _filter_context(self, agent: Agent) -> List[Document]:
        """Filter and enhance context using knowledge graph insights."""
        filtered_context = agent.context.copy()
        
        if self.knowledge_graph is not None:
            # Extract entities from question for subgraph search
            question_entities = self._extract_entities(agent.question)
            
            # Find relevant subgraphs for each entity
            kg_insights = []
            for entity in question_entities:
                subgraph_content = self._get_entity_subgraph(entity)
                if subgraph_content:
                    kg_insights.append(subgraph_content)
            
            # Add knowledge graph insights if found
            if kg_insights:
                kg_context = Document(page_content="\n".join([
                    "Knowledge Graph Context:",
                    *kg_insights,
                    "\nConsider these relationships in your solution."
                ]))
                filtered_context.append(kg_context)
            
            # Get relevant template based on graph structure
            template = self._get_context_template(question_entities)
            if template:
                filtered_context.append(Document(
                    page_content=f"Solution Framework:\n{template}"
                ))
        
        # Add parent solution with performance analysis
        if agent.parent and agent.parent.solution:
            solution_context = self._analyze_parent_solution(agent.parent)
            filtered_context.append(Document(page_content=solution_context))
        
        return filtered_context
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text using graph node matching."""
        entities = set()
        if self.knowledge_graph:
            # Convert text to lowercase for matching
            text_lower = text.lower()
            # Sort nodes by length (longest first) to match most specific entities
            nodes = sorted(self.knowledge_graph.nodes(), key=lambda x: len(str(x)), reverse=True)
            for node in nodes:
                if str(node).lower() in text_lower:
                    entities.add(str(node))
        return list(entities)
    
    def _get_entity_subgraph(self, entity: str) -> str:
        """Extract relevant information from entity's neighborhood in knowledge graph."""
        if not self.knowledge_graph:
            return ""
            
        # Find matching node
        matching_nodes = [n for n in self.knowledge_graph.nodes()
                         if str(n).lower() == entity.lower()]
        if not matching_nodes:
            return ""
            
        node = matching_nodes[0]
        subgraph = nx.ego_graph(self.knowledge_graph, node, radius=2)
        
        # Extract relationships and properties
        insights = []
        
        # Node properties
        node_attrs = self.knowledge_graph.nodes[node]
        if node_attrs.get('description'):
            insights.append(f"• {entity}: {node_attrs['description']}")
            
        # Edge relationships
        for src, dst, data in subgraph.edges(data=True):
            rel = data.get('relationship', 'relates to')
            insights.append(f"• {src} {rel} {dst}")
            
        return "\n".join(insights)
    
    def _get_context_template(self, entities: List[str]) -> str:
        """Generate context-aware template based on knowledge graph patterns."""
        if not self.knowledge_graph or not entities:
            return ""
            
        # Analyze subgraph structure to identify pattern
        pattern_nodes = set()
        for entity in entities:
            matching_nodes = [n for n in self.knowledge_graph.nodes()
                            if str(n).lower() == entity.lower()]
            if matching_nodes:
                pattern_nodes.update(nx.neighbors(self.knowledge_graph, matching_nodes[0]))
        
        # Generate template sections based on node types and relationships
        sections = []
        if pattern_nodes:
            # Group nodes by type/category
            categories = {}
            for node in pattern_nodes:
                category = self.knowledge_graph.nodes[node].get('category', 'general')
                if category not in categories:
                    categories[category] = []
                categories[category].append(str(node))
            
            # Create template sections
            for category, nodes in categories.items():
                section_title = category.replace('_', ' ').title()
                points = [f"   • Impact on {node}" for node in nodes]
                sections.append(f"{section_title} Considerations:\n" + "\n".join(points))
        
        if not sections:
            return ""
            
        return "Analysis Template:\n\n" + "\n\n".join(sections)
    
    def _analyze_parent_solution(self, parent: Agent) -> str:
        """Analyze parent solution for insights and improvements."""
        return f"""Previous Solution Analysis:
Score: {parent.score}/10, Confidence: {parent.confidence:.1%}

Solution Attempt:
{parent.solution}

Key Validation Points:
{parent.validation}

Improvement Areas:
{self._extract_improvement_areas(parent.validation)}
"""
    
    def _extract_improvement_areas(self, validation: str) -> str:
        """Extract improvement suggestions from validation text."""
        improvements = []
        for line in validation.split('\n'):
            line = line.strip()
            # Look for constructive feedback patterns
            if any(x in line.lower() for x in ['should', 'could', 'missing', 'lacks', 'needs', 'consider']):
                improvements.append(f"• {line}")
        return "\n".join(improvements) if improvements else "No specific improvements identified."
    
    def backpropagate(self, agent: Agent):
        """Propagate rewards up the tree with improved confidence handling."""
        current = agent
        while current.parent is not None:
            parent = current.parent
            
            # Confidence-weighted reward with minimum baseline
            agent_reward = max(agent.score * agent.confidence, 1.0)  # Minimum reward to avoid over-penalization
            
            # Update parent statistics with exponential moving average
            alpha = 0.8  # Weight for new reward vs historical average
            parent.visit_count += 1
            parent.avg_reward = (
                alpha * ((agent_reward + parent.avg_reward * parent.visit_count) /
                        (parent.visit_count + 1)) +
                (1 - alpha) * parent.avg_reward
            )
            
            current = parent
    
    def solve(self) -> Agent:
        """Main solving process with MCTS."""
        # Initial generation for root
        self.root.generate_solution()
        self.root.validate()
        self.root.assess()
        
        # Run expansions
        for _ in range(self.max_expansion):
            selected = self.select_agent()
            self.expand(selected)
            
            # Check for successful terminal agents
            terminal_passed = [
                a for a in self.agents.values()
                if a.terminal and a.passed
            ]
            if terminal_passed:
                # Return highest scoring passed terminal agent
                return max(terminal_passed, key=lambda a: a.score)
        
        # If no terminal solution, return best overall
        return max(self.agents.values(), key=lambda a: a.score)
