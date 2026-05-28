import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { ComparisonTile } from '../components/ComparisonTile';
import { DequePanel } from '../components/DequePanel';
import { MapCanvas } from '../components/MapCanvas';
import { RouteTable } from '../components/RouteTable';
import type { AlgorithmRun, Dataset, Depot, Route } from '../types';
import { getDatasetMeta, runEnhanced } from '../services/deliveryApi';

type StoredRunSummary = {
  id: string;
  algorithm: string;
  datasetId: string;
  kpis: AlgorithmRun['kpis'];
  representatives: AlgorithmRun['representatives'];
};

const DATASET_STORAGE_KEY = 'uploadedDatasetFileMeta';
const BASELINE_STORAGE_KEY = 'baselineRunSummary';
const ENHANCED_STORAGE_KEY = 'enhancedRunSummary';

const formatSalesRepName = (repId?: string | null) => {
  if (!repId) return '';
  return repId.replace('-AGE-', '-');
};

const EnhancedRun: React.FC = () => {
  const navigate = useNavigate();

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [depot, setDepot] = useState<Depot | null>(null);
  const [baselineSummary, setBaselineSummary] = useState<StoredRunSummary | null>(null);
  const [enhancedRun, setEnhancedRun] = useState<AlgorithmRun | null>(null);

  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');

  const [parameters, setParameters] = useState({
    alphaWeight: '0.60',
    betaWeight: '0.40',
    maxIterations: '30',
    borderFraction: '0.70',
  });

  const [showLabels, setShowLabels] = useState(true);
  const [showRouteNumbers, setShowRouteNumbers] = useState(false);
  const [showAllRoutes, setShowAllRoutes] = useState(false);
  const [selectedRouteId, setSelectedRouteId] = useState('');

  useEffect(() => {
    const storedDataset = localStorage.getItem(DATASET_STORAGE_KEY);
    const storedBaseline = localStorage.getItem(BASELINE_STORAGE_KEY);

    if (storedDataset) {
      setDataset(JSON.parse(storedDataset));
    }

    if (storedBaseline) {
      setBaselineSummary(JSON.parse(storedBaseline));
    }
  }, []);

  useEffect(() => {
    const loadMeta = async () => {
      if (!dataset?.id) return;
      try {
        const meta = await getDatasetMeta(dataset.id);
        if (meta.depot) {
          setDepot(meta.depot);
        }
      } catch {
        // ignore for now; map can still render without depot
      }
    };

    loadMeta();
  }, [dataset]);

  const routeOptions = useMemo(() => {
    if (!enhancedRun) return [];
    return enhancedRun.routes.map((route) => ({
      value: route.id,
      label: `${formatSalesRepName(route.representativeName)} (${route.stops.length} stops)`,
    }));
  }, [enhancedRun]);

  const selectedRoutesForDisplay: Route[] = useMemo(() => {
    if (!enhancedRun) return [];
    if (showAllRoutes) return enhancedRun.routes;
    if (!selectedRouteId) return enhancedRun.routes;
    const match = enhancedRun.routes.find((route) => route.id === selectedRouteId);
    return match ? [match] : enhancedRun.routes;
  }, [enhancedRun, selectedRouteId, showAllRoutes]);

  const displayDepot: Depot | null = useMemo(() => {
    if (
      enhancedRun?.previewSummary?.depotLat != null &&
      enhancedRun?.previewSummary?.depotLon != null
    ) {
      return {
        id: 'PREVIEW-DEPOT',
        name: 'Depot',
        lat: enhancedRun.previewSummary.depotLat,
        lon: enhancedRun.previewSummary.depotLon,
      };
    }
    return depot;
  }, [enhancedRun, depot]);

  const highlightedNodes = useMemo(() => {
    if (showAllRoutes) return [];
    const selected = enhancedRun?.routes.find((route) => route.id === selectedRouteId);
    return selected ? selected.stops.map((stop) => stop.nodeId) : [];
  }, [enhancedRun, selectedRouteId, showAllRoutes]);

  const selectedRoute = useMemo(() => {
    if (!enhancedRun || !selectedRouteId) return null;
    return enhancedRun.routes.find((route) => route.id === selectedRouteId) ?? null;
  }, [enhancedRun, selectedRouteId]);

  const calculateDelta = (enhanced: number, baseline: number): number => {
    if (!baseline) return 0;
    return Number((((enhanced - baseline) / baseline) * 100).toFixed(1));
  };

  const handleRun = async () => {
    if (!dataset || !baselineSummary) {
      setMessage('Baseline run is required before enhanced execution.');
      return;
    }

    try {
      setBusy(true);
      setMessage('Running enhanced DEQ experiment on backend...');

      const result = await runEnhanced({
        datasetId: dataset.id,
        baselineRunId: baselineSummary.id,
        alphaWeight: Number(parameters.alphaWeight) || 0.6,
        betaWeight: Number(parameters.betaWeight) || 0.4,
        maxIterations: Number(parameters.maxIterations) || 30,
        borderFraction: Number(parameters.borderFraction) || 0.7,
        runProfile:
          dataset.datasetRole === 'primary_reconstruction'
            ? 'amazon_expanded_search'
            : dataset.datasetRole === 'comparative_template'
              ? 'zomato_expanded_search'
              : 'default_balanced',
      });

      const summary: StoredRunSummary = {
        id: result.id,
        algorithm: result.algorithm,
        datasetId: result.datasetId,
        kpis: result.kpis,
        representatives: result.representatives,
      };

      localStorage.setItem(ENHANCED_STORAGE_KEY, JSON.stringify(summary));
      setEnhancedRun(result);
      setSelectedRouteId(result.routes[0]?.id ?? '');
      setMessage('Enhanced run completed.');
    } catch (err) {
      setMessage(err instanceof Error ? err.message : 'Enhanced run failed.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="h-screen flex flex-col">
        <div className="bg-white border-b border-gray-200 px-8 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Enhanced Route Optimization</h1>
            <p className="text-sm text-gray-600">
              Greedy Nearest Neighbor + Double-Ended Queue Rebalancing
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Button variant="secondary" onClick={() => navigate('/baseline')}>
              Back to Baseline
            </Button>
            <Button
              variant="secondary"
              onClick={() => navigate('/compare')}
              disabled={!enhancedRun}
            >
              Proceed to Compare
            </Button>
          </div>
        </div>

        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 p-6 overflow-hidden min-w-0 flex flex-col">
            <div className="flex flex-col gap-6 h-full">
              <div className="flex-1 min-h-[480px]">
                {enhancedRun ? (
                  <MapCanvas
                    routes={selectedRoutesForDisplay}
                    depot={displayDepot}
                    showLabels={showLabels}
                    showRouteNumbers={showRouteNumbers}
                    highlightedNodes={highlightedNodes}
                  />
                ) : (
                  <Card className="h-full flex items-center justify-center">
                    <div className="text-center flex flex-col items-center justify-center gap-4 max-w-xl">
                      <p className="text-gray-900 text-xl font-semibold">
                        Run the enhanced algorithm with DEQ rebalancing
                      </p>
                      <p className="text-sm text-gray-600">
                        This stage starts from the baseline solution, then improves route assignment
                        using priority scoring based on time difference and rating.
                      </p>
                      <Button onClick={handleRun} disabled={busy || !baselineSummary}>
                        {busy ? 'Running...' : 'Run Enhanced Algorithm'}
                      </Button>
                    </div>
                  </Card>
                )}
              </div>

              {enhancedRun && (
                <RouteTable
                  routes={selectedRoute ? [selectedRoute] : enhancedRun.routes}
                  title={
                    selectedRoute
                      ? `Route Details — ${formatSalesRepName(selectedRoute.representativeName)}`
                      : 'All Representative Routes'
                  }
                />
              )}
            </div>
          </div>

          <div className="w-[420px] bg-white border-l border-gray-200 p-6 overflow-auto">
            {!enhancedRun ? (
              <div className="space-y-6">
                <Card>
                  <h2 className="text-sm font-semibold text-gray-900 mb-4">
                    Enhanced Algorithm Setup
                  </h2>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-2">
                        Time Difference Weight (α)
                      </label>
                      <input
                        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                        value={parameters.alphaWeight}
                        onChange={(e) =>
                          setParameters((prev) => ({
                            ...prev,
                            alphaWeight: e.target.value,
                          }))
                        }
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-2">
                        Rating Weight (β)
                      </label>
                      <input
                        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                        value={parameters.betaWeight}
                        onChange={(e) =>
                          setParameters((prev) => ({
                            ...prev,
                            betaWeight: e.target.value,
                          }))
                        }
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      {/* <div>
                        <label className="block text-xs font-medium text-gray-600 mb-2">
                          Max Iterations
                        </label>
                        <input
                          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                          value={parameters.maxIterations}
                          onChange={(e) =>
                            setParameters((prev) => ({
                              ...prev,
                              maxIterations: e.target.value,
                            }))
                          }
                        />
                      </div>

                      <div>
                        <label className="block text-xs font-medium text-gray-600 mb-2">
                          Border Fraction
                        </label>
                        <input
                          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                          value={parameters.borderFraction}
                          onChange={(e) =>
                            setParameters((prev) => ({
                              ...prev,
                              borderFraction: e.target.value,
                            }))
                          }
                        />
                      </div> */}
                    </div>

                    <Button
                      onClick={handleRun}
                      disabled={busy || !baselineSummary}
                      className="w-full"
                    >
                      {busy ? 'Running...' : 'Run Enhanced Algorithm'}
                    </Button>
                  </div>
                </Card>

                <Card>
                  <h2 className="text-sm font-semibold text-gray-900 mb-3">Run Context</h2>
                  <div className="space-y-2 text-sm text-gray-600">
                    <div>
                      <span className="font-medium text-gray-800">Dataset:</span>{' '}
                      {dataset?.name ?? dataset?.id ?? 'Not loaded'}
                    </div>
                    <div>
                      <span className="font-medium text-gray-800">Baseline Run:</span>{' '}
                      {baselineSummary?.id ?? 'Required'}
                    </div>
                  </div>
                </Card>

                {message && (
                  <Card>
                    <p className="text-sm text-gray-700">{message}</p>
                  </Card>
                )}
              </div>
            ) : (
              <div className="space-y-6">
                {message && (
                  <Card>
                    <p className="text-sm text-gray-700">{message}</p>
                  </Card>
                )}

                {baselineSummary && (
                  <div>
                    <h2 className="text-sm font-semibold text-gray-900 mb-3">
                      Comparison vs Baseline
                    </h2>
                    <div className="space-y-3">
                      <ComparisonTile
                        label="Total Distance"
                        baselineValue={baselineSummary.kpis.totalDistance}
                        enhancedValue={enhancedRun.kpis.totalDistance}
                        unit=" km"
                        lowerIsBetter
                      />
                      <ComparisonTile
                        label="Travel Time"
                        baselineValue={baselineSummary.kpis.travelTime}
                        enhancedValue={enhancedRun.kpis.travelTime}
                        unit=" hr"
                        lowerIsBetter
                      />
                      <ComparisonTile
                        label="Operational Time"
                        baselineValue={baselineSummary.kpis.operationalTime}
                        enhancedValue={enhancedRun.kpis.operationalTime}
                        unit=" hr"
                        lowerIsBetter
                      />
                      <ComparisonTile
                        label="Fairness"
                        baselineValue={baselineSummary.kpis.fairness}
                        enhancedValue={enhancedRun.kpis.fairness}
                      />
                    </div>
                  </div>
                )}

                <Card>
                  <h2 className="text-sm font-semibold text-gray-900 mb-4">Map Controls</h2>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-2">
                        Route Filter
                      </label>
                      <select
                        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white"
                        value={selectedRouteId}
                        onChange={(e) => {
                          setSelectedRouteId(e.target.value);
                          setShowAllRoutes(false);
                        }}
                        disabled={!enhancedRun}
                      >
                        <option value="">All routes</option>
                        {routeOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={showAllRoutes}
                        onChange={(e) => setShowAllRoutes(e.target.checked)}
                      />
                      Show all routes
                    </label>

                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={showLabels}
                        onChange={(e) => setShowLabels(e.target.checked)}
                      />
                      Show Customer Labels and Assigned Reps
                    </label>

                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={showRouteNumbers}
                        onChange={(e) => setShowRouteNumbers(e.target.checked)}
                      />
                      Show route numbers
                    </label>
                  </div>
                </Card>

                <Card>
                  <h2 className="text-sm font-semibold text-gray-900 mb-3">Enhanced KPI Summary</h2>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-slate-50 p-3">
                      <p className="text-xs text-gray-500">Distance</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {enhancedRun.kpis.totalDistance.toFixed(2)} km
                      </p>
                    </div>
                    <div className="rounded-lg bg-slate-50 p-3">
                      <p className="text-xs text-gray-500">Travel Time</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {enhancedRun.kpis.travelTime.toFixed(2)} hr
                      </p>
                    </div>
                    <div className="rounded-lg bg-slate-50 p-3">
                      <p className="text-xs text-gray-500">Operational Time</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {enhancedRun.kpis.operationalTime.toFixed(2)} hr
                      </p>
                    </div>
                    <div className="rounded-lg bg-slate-50 p-3">
                      <p className="text-xs text-gray-500">Fairness</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {enhancedRun.kpis.fairness.toFixed(3)}
                      </p>
                    </div>
                    <div className="rounded-lg bg-slate-50 p-3">
                      <p className="text-xs text-gray-500">CPU Runtime</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {enhancedRun.cpuRuntimeSeconds !== undefined
                          ? enhancedRun.cpuRuntimeSeconds.toFixed(3)
                          : '—'}{' '}
                        sec
                      </p>
                    </div>
                  </div>
                </Card>

                <DequePanel representatives={enhancedRun.representatives} />

                <Card>
                  <h2 className="text-sm font-semibold text-gray-900 mb-3">Quick Result Notes</h2>
                  <div className="space-y-2 text-sm text-gray-600">
                    <p>
                      Distance change:{' '}
                      <span className="font-medium text-gray-800">
                        {calculateDelta(
                          enhancedRun.kpis.totalDistance,
                          baselineSummary?.kpis.totalDistance ?? enhancedRun.kpis.totalDistance
                        )}
                        %
                      </span>
                    </p>
                    <p>
                      Travel time change:{' '}
                      <span className="font-medium text-gray-800">
                        {calculateDelta(
                          enhancedRun.kpis.travelTime,
                          baselineSummary?.kpis.travelTime ?? enhancedRun.kpis.travelTime
                        )}
                        %
                      </span>
                    </p>
                    <p>
                      Fairness change:{' '}
                      <span className="font-medium text-gray-800">
                        {calculateDelta(
                          enhancedRun.kpis.fairness,
                          baselineSummary?.kpis.fairness ?? enhancedRun.kpis.fairness
                        )}
                        %
                      </span>
                    </p>
                  </div>
                </Card>

                <Button onClick={() => navigate('/compare')} className="w-full">
                  Proceed to Comparison View
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedRun;
