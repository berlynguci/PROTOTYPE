export interface Coordinate {
  lat: number;
  lon: number;
}

export interface Depot {
  id: string;
  lat: number;
  lon: number;
  name?: string;
}

export interface Customer {
  id: string;
  lat: number;
  lon: number;
  name?: string;
  orderId?: string;
}

export interface AddedCustomer {
  id: string;
  label: string;
  lat: number;
  lon: number;
  address?: string;
  assignedRep?: string;
  customerNumber?: number;
}

export interface RouteStop {
  stopNumber: number;
  nodeId: string;
  nodeName: string;
  legDistance: number;
  cumulativeDistance: number;
  eta: number;
  lat: number;
  lon: number;
  orderId?: string;
  predictedEtaMin?: number;

  // new: visual-only geometry from previous point to this stop
  legPath?: Coordinate[];
}

export interface Route {
  id: string;
  representativeId: string;
  representativeName: string;
  stops: RouteStop[];
  color: string;
  returnLegDistance?: number;
  routeDistance?: number;

  // new: visual-only geometry from last stop back to depot
  returnPath?: Coordinate[];
}

export interface KPIMetrics {
  totalDistance: number;
  // legacy / UI fields used across pages
  totalTime: number;
  travelTime: number;
  operationalTime: number;
  computeTime: number;
  fairness: number;
  workloadBalance: number;
  coverage: number;
  scalability: number;

  // additional fields referenced by pages_v1 mock data and components
  numberOfStops: number;
  delayScore: number;
  ratingPenalty: number;
  coverageRatio: number;
  workloadBalanceIndex: number;
  jainsFairnessIndex: number;

  avgTotalDistance: number;
  avgTravelTime: number;
}

export interface Representative {
  id: string;
  name: string;
  workload: number;
  opportunityScore: number;
  priorityScore: number;
  queuePosition: number;
  assignedCustomers: number;
  color?: string;

  totalDistance?: number;
  totalTime?: number;
}

export interface Dataset {
  id: string;
  name: string;
  rows: number;
  depots: number;
  customers: number;
  orders: number;
  datasetRole?: 'primary_reconstruction' | 'comparative_template' | 'generic_uploaded_dataset';
  sourceLabel?: string;
  reconstructedBaselineReady?: boolean;
  reconstructedBaselineName?: string;
}

export interface ValidationSummary {
  records: number;
  depots: number;
  customers: number;
  orders: number;
  avgRating?: number;
}

export interface ValidationResult {
  isValid: boolean;
  invalidCoordinates: number;
  duplicateRows: number;
  nearDuplicates: number;
  summary: ValidationSummary;
}

export interface TrainingMetrics {
  mae: number;
  rmse: number;
  r2: number;
}

export interface PreviewSummary {
  selectionStrategy?: string;
  maxRoutes?: number;
  maxTotalStops?: number;
  maxDistanceFromDepotKm?: number;
  depotLat?: number | null;
  depotLon?: number | null;
}

export interface RunLogEntry {
  iteration: number;
  moved_order?: string;
  from_rep: string;
  to_rep: string;
  fairness_before?: number;
  fairness_after?: number;
  distance_before?: number;
  distance_after?: number;
  operational_before?: number;
  operational_after?: number;
  score?: number;
  score_before?: number;
  score_after?: number;
  score_gain?: number;
  fairness_gain?: number;
  distance_gain?: number;
  time_gain?: number;
  accepted: boolean;
  reason?: string;
}

export interface AlgorithmRun {
  id: string;
  name: string;
  algorithm: string;
  datasetId: string;
  runType: 'baseline' | 'enhanced';
  datasetRole?: 'primary_reconstruction' | 'comparative_template' | 'generic_uploaded_dataset';
  sourceLabel?: string;
  routes: Route[];
  kpis: KPIMetrics;
  representatives: Representative[];
  trainingMetrics: TrainingMetrics;
  trainingComparison?: {
    baseline: TrainingMetrics;
    enhanced: TrainingMetrics;
  };
  notes?: string[];
  baselineRunId?: string;
  runLog?: RunLogEntry[];
  previewSummary?: PreviewSummary;
  previewMode?: boolean;
  matrixMode?: string;
  matrixStats?: {
    previewPoints: number;
    matrixPairs: number;
  };

  cpuRuntimeSeconds?: number;
  cpuRuntimeLabel?: string;
}

export interface BaselineRunRequest {
  datasetId: string;
  numRepresentatives: number;
  avgSpeedKmph: number;
  serviceMinutesPerStop: number;
  seed: number;
  runProfile?: 'default_balanced' | 'amazon_expanded_search' | 'zomato_expanded_search';
}

export interface EnhancedRunRequest {
  datasetId: string;
  baselineRunId: string;
  alphaWeight?: number;
  betaWeight?: number;
  maxIterations?: number;
  borderFraction?: number;
  runProfile?: 'default_balanced' | 'amazon_expanded_search' | 'zomato_expanded_search';
}
