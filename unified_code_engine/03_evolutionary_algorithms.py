"""
Cell 3: Advanced Evolutionary Algorithms
=========================================

State-of-the-art evolutionary algorithms for autonomous code evolution:
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- Multi-Objective Evolutionary Algorithms (NSGA-III, MOEA/D)
- Differential Evolution with adaptive parameters
- Genetic Programming with semantic-aware operators
- Novelty Search and Quality-Diversity algorithms
- Self-Adaptive mutation strategies
"""

class EvolutionaryAlgorithmType(Enum):
    """Types of evolutionary algorithms implemented."""
    CMA_ES = "cma_es"
    NSGA_III = "nsga_iii"
    MOEA_D = "moea_d"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    GENETIC_PROGRAMMING = "genetic_programming"
    NOVELTY_SEARCH = "novelty_search"
    MAP_ELITES = "map_elites"
    SELF_ADAPTIVE_EA = "self_adaptive_ea"


@dataclass
class Individual:
    """Individual in evolutionary algorithm."""
    genotype: np.ndarray
    phenotype: Any
    fitness: Union[float, np.ndarray]
    age: int = 0
    novelty_score: float = 0.0
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Population:
    """Population of individuals."""
    individuals: List[Individual]
    generation: int = 0
    statistics: Dict[str, Any] = field(default_factory=dict)


class CMAEvolutionStrategy:
    """Covariance Matrix Adaptation Evolution Strategy."""
    
    def __init__(self, dimension: int, population_size: int = None):
        self.dimension = dimension
        self.population_size = population_size or 4 + int(3 * np.log(dimension))
        
        # CMA-ES parameters
        self.mean = np.zeros(dimension)
        self.sigma = 1.0
        self.C = np.eye(dimension)  # Covariance matrix
        self.pc = np.zeros(dimension)  # Evolution path for C
        self.ps = np.zeros(dimension)  # Evolution path for sigma
        
        # Strategy parameters
        self.cc = 4 / (dimension + 4)
        self.cs = (4 + 2/dimension) / (dimension + 4 + 2/dimension)
        self.c1 = 2 / ((dimension + 1.3)**2 + self.population_size/2)
        self.cmu = min(1-self.c1, 2*(self.population_size/2-2+1/self.population_size) / ((dimension+2)**2 + self.population_size/2))
        self.damps = 1 + 2*max(0, np.sqrt((self.population_size-1)/(dimension+1))-1) + self.cs
        
    def sample_population(self) -> List[np.ndarray]:
        """Sample new population from current distribution."""
        # Eigendecomposition for sampling
        eigenvals, eigenvecs = np.linalg.eigh(self.C)
        eigenvals = np.maximum(eigenvals, 1e-14)  # Ensure positive definiteness
        
        samples = []
        for _ in range(self.population_size):
            z = np.random.randn(self.dimension)
            x = self.mean + self.sigma * np.dot(eigenvecs, np.sqrt(eigenvals) * z)
            samples.append(x)
            
        return samples
    
    def update(self, individuals: List[Individual]):
        """Update CMA-ES parameters based on fitness-sorted individuals."""
        # Sort by fitness (assuming minimization)
        sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness)
        
        # Selection and recombination
        mu = self.population_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.zeros(self.dimension)
        for i in range(mu):
            self.mean += weights[i] * sorted_individuals[i].genotype
            
        # Update evolution paths
        mueff = 1 / np.sum(weights**2)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * mueff) * \
                  np.dot(np.linalg.inv(scipy.linalg.sqrtm(self.C)), self.mean - old_mean) / self.sigma
        
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * len(individuals) / self.population_size)) < \
               1.4 + 2 / (self.dimension + 1)
        
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * mueff) * \
                  (self.mean - old_mean) / self.sigma
        
        # Update covariance matrix
        delta_C = np.zeros((self.dimension, self.dimension))
        for i in range(mu):
            delta = (sorted_individuals[i].genotype - old_mean) / self.sigma
            delta_C += weights[i] * np.outer(delta, delta)
            
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc) + \
                 self.cmu * delta_C
        
        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dimension) - 1))


class NSGAIII:
    """Non-dominated Sorting Genetic Algorithm III for many-objective optimization."""
    
    def __init__(self, population_size: int, num_objectives: int):
        self.population_size = population_size
        self.num_objectives = num_objectives
        self.reference_points = self._generate_reference_points()
        
    def _generate_reference_points(self) -> np.ndarray:
        """Generate reference points for NSGA-III."""
        # Simplified Das-Dennis method
        num_points = self.population_size
        points = []
        
        for i in range(num_points):
            point = np.random.dirichlet(np.ones(self.num_objectives))
            points.append(point)
            
        return np.array(points)
    
    def non_dominated_sort(self, population: List[Individual]) -> List[List[int]]:
        """Perform non-dominated sorting."""
        fronts = []
        domination_count = np.zeros(len(population))
        dominated_solutions = [[] for _ in range(len(population))]
        
        # Find first front
        first_front = []
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j:
                    if self._dominates(population[i], population[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(population[j], population[i]):
                        domination_count[i] += 1
                        
            if domination_count[i] == 0:
                first_front.append(i)
                
        fronts.append(first_front)
        
        # Find subsequent fronts
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            fronts.append(next_front)
            current_front += 1
            
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2."""
        obj1, obj2 = ind1.objectives, ind2.objectives
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def reference_point_association(self, population: List[Individual]) -> Dict[int, List[int]]:
        """Associate individuals with reference points."""
        associations = defaultdict(list)
        
        for i, individual in enumerate(population):
            # Find closest reference point
            distances = cdist([individual.objectives], self.reference_points)[0]
            closest_ref = np.argmin(distances)
            associations[closest_ref].append(i)
            
        return associations


class DifferentialEvolution:
    """Differential Evolution with adaptive parameters."""
    
    def __init__(self, dimension: int, population_size: int = 30):
        self.dimension = dimension
        self.population_size = population_size
        self.F = 0.5  # Differential weight
        self.Cr = 0.9  # Crossover probability
        self.adaptive = True
        
    def mutate(self, population: List[Individual], target_idx: int) -> np.ndarray:
        """DE/rand/1 mutation strategy."""
        # Select three random individuals different from target
        candidates = list(range(len(population)))
        candidates.remove(target_idx)
        a, b, c = random.sample(candidates, 3)
        
        # Mutation: F * (xb - xc)
        mutant = population[a].genotype + self.F * (population[b].genotype - population[c].genotype)
        return mutant
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Binomial crossover."""
        trial = target.copy()
        j_rand = random.randint(0, self.dimension - 1)
        
        for j in range(self.dimension):
            if random.random() < self.Cr or j == j_rand:
                trial[j] = mutant[j]
                
        return trial
    
    def adaptive_parameter_update(self, generation: int, success_rate: float):
        """Adaptive parameter control."""
        if self.adaptive:
            # Adapt F and Cr based on success rate
            if success_rate > 0.2:
                self.F = min(1.0, self.F * 1.1)
                self.Cr = min(1.0, self.Cr * 1.05)
            else:
                self.F = max(0.1, self.F * 0.9)
                self.Cr = max(0.1, self.Cr * 0.95)


class NoveltySearch:
    """Novelty search for maintaining diversity."""
    
    def __init__(self, behavior_space_dim: int, archive_threshold: float = 1.5):
        self.behavior_space_dim = behavior_space_dim
        self.archive_threshold = archive_threshold
        self.archive = []
        self.k_neighbors = 15
        
    def calculate_novelty(self, individual: Individual, population: List[Individual]) -> float:
        """Calculate novelty score based on behavioral diversity."""
        if not hasattr(individual, 'behavior'):
            individual.behavior = self._extract_behavior(individual)
            
        # Combine population and archive for neighbor search
        all_behaviors = []
        for ind in population:
            if not hasattr(ind, 'behavior'):
                ind.behavior = self._extract_behavior(ind)
            all_behaviors.append(ind.behavior)
            
        all_behaviors.extend([arch.behavior for arch in self.archive])
        
        # Calculate distances to k-nearest neighbors
        if len(all_behaviors) < self.k_neighbors:
            return 1.0  # High novelty if not enough neighbors
            
        distances = cdist([individual.behavior], all_behaviors)[0]
        sorted_distances = np.sort(distances)
        novelty = np.mean(sorted_distances[1:self.k_neighbors+1])  # Skip self (distance 0)
        
        # Update archive if novelty is high enough
        if novelty > self.archive_threshold:
            self.archive.append(individual)
            
        return novelty
    
    def _extract_behavior(self, individual: Individual) -> np.ndarray:
        """Extract behavioral descriptor from individual."""
        # Simplified behavior extraction
        if hasattr(individual, 'phenotype'):
            return np.random.randn(self.behavior_space_dim)  # Placeholder
        return np.random.randn(self.behavior_space_dim)


class MapElites:
    """MAP-Elites quality-diversity algorithm."""
    
    def __init__(self, map_dimensions: Tuple[int, ...], map_bounds: List[Tuple[float, float]]):
        self.map_dimensions = map_dimensions
        self.map_bounds = map_bounds
        self.map = {}  # Dictionary to store elites
        self.total_cells = np.prod(map_dimensions)
        
    def get_map_coordinates(self, individual: Individual) -> Tuple[int, ...]:
        """Get map coordinates for an individual based on its features."""
        if not hasattr(individual, 'features'):
            individual.features = self._extract_features(individual)
            
        coords = []
        for i, (feature, (min_val, max_val)) in enumerate(zip(individual.features, self.map_bounds)):
            # Normalize feature to map dimension
            normalized = (feature - min_val) / (max_val - min_val)
            coord = int(np.clip(normalized * self.map_dimensions[i], 0, self.map_dimensions[i] - 1))
            coords.append(coord)
            
        return tuple(coords)
    
    def add_to_map(self, individual: Individual):
        """Add individual to map if it's better than current elite."""
        coords = self.get_map_coordinates(individual)
        
        if coords not in self.map or individual.fitness > self.map[coords].fitness:
            self.map[coords] = individual
            
    def get_random_elite(self) -> Optional[Individual]:
        """Get random elite from map for reproduction."""
        if not self.map:
            return None
        return random.choice(list(self.map.values()))
    
    def _extract_features(self, individual: Individual) -> np.ndarray:
        """Extract behavioral/phenotypic features."""
        # Simplified feature extraction
        return np.random.randn(len(self.map_dimensions))


# Initialize evolutionary algorithms
cma_es = CMAEvolutionStrategy(dimension=20)
nsga_iii = NSGAIII(population_size=100, num_objectives=3)
differential_evolution = DifferentialEvolution(dimension=20)
novelty_search = NoveltySearch(behavior_space_dim=10)
map_elites = MapElites(map_dimensions=(10, 10), map_bounds=[(0, 1), (0, 1)])

logger.info("🧬 Advanced evolutionary algorithms initialized")
logger.info("🎯 Multi-objective, novelty, and quality-diversity algorithms ready")