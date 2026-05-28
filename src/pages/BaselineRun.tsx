import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeftIcon,
  ClockIcon,
  MapIcon,
  PlayIcon,
  PlusIcon,
  RouteIcon,
  UsersIcon,
} from 'lucide-react';

import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Input } from '../components/Input';
import { KPICard } from '../components/KPICard';
import { MapCanvas } from '../components/MapCanvas';
import AddCustomerModal, { type AddCustomer } from '../components/AddCustomerModal';
import { RouteTable } from '../components/RouteTable';
import { Select } from '../components/Select';

import type { AddedCustomer, AlgorithmRun, Dataset, Depot, Route } from '../types';
import { addCustomersToBaseline, getDatasetMeta, runBaseline } from '../services/deliveryApi';

type StoredRunSummary = {
  id: string;
  algorithm: string;
  datasetId: string;
  kpis: AlgorithmRun['kpis'];
  representatives: AlgorithmRun['representatives'];
};

const BASELINE_STORAGE_KEY = 'baselineRunSummary';
const DATASET_STORAGE_KEY = 'uploadedDatasetFileMeta';

const formatSalesRepName = (repId?: string | null) => {
  if (!repId) return '';
  return repId.replace('-AGE-', '-');
};

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const toRad = (v: number) => (v * Math.PI) / 180;
  const R = 6371;

  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);

  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function extractCustomerNumber(name: string): number | null {
  const match = name.match(/Customer\s+(\d+)/i);
  if (!match) return null;
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
}

function getNextCustomerNumber(run: AlgorithmRun | null, addedCustomers: AddedCustomer[]): number {
  let maxNumber = 0;

  if (run) {
    for (const route of run.routes) {
      for (const stop of route.stops) {
        const n = extractCustomerNumber(stop.nodeName);
        if (n != null && n > maxNumber) maxNumber = n;
      }
    }
  }

  for (const customer of addedCustomers) {
    if (customer.customerNumber && customer.customerNumber > maxNumber) {
      maxNumber = customer.customerNumber;
    } else {
      const n = extractCustomerNumber(customer.label);
      if (n != null && n > maxNumber) maxNumber = n;
    }
  }

  return maxNumber + 1;
}

function applyBackendAddedCustomerAssignments(
  updatedRun: AlgorithmRun,
  customers: AddedCustomer[]
): AddedCustomer[] {
  const repByCustomerNumber = new Map<number, string>();
  const repByLabel = new Map<string, string>();

  for (const route of updatedRun.routes) {
    for (const stop of route.stops) {
      const orderId = (stop as { orderId?: string }).orderId;
      if (typeof orderId === 'string' && orderId.startsWith('ADDED-ORDER-')) {
        const customerNumber = extractCustomerNumber(stop.nodeName);
        if (customerNumber != null) {
          repByCustomerNumber.set(customerNumber, route.representativeName);
        }
        repByLabel.set(stop.nodeName, route.representativeName);
      }
    }
  }

  return customers.map((customer) => {
    const assignedRep =
      (customer.customerNumber != null
        ? repByCustomerNumber.get(customer.customerNumber)
        : undefined) ??
      repByLabel.get(customer.label) ??
      customer.assignedRep ??
      '';

    return {
      ...customer,
      assignedRep,
      label: assignedRep ? `Customer ${customer.customerNumber} - ${assignedRep}` : customer.label,
    };
  });
}

const BaselineRun: React.FC = () => {
  const navigate = useNavigate();

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [depot, setDepot] = useState<Depot | null>(null);
  const [run, setRun] = useState<AlgorithmRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');

  const [parameters, setParameters] = useState({
    numRepresentatives: '4',
    speed: '40',
    serviceMinutes: '8',
    seed: '42',
  });

  const [addedCustomers, setAddedCustomers] = useState<AddedCustomer[]>([]);

  const [showLabels, setShowLabels] = useState(true);
  const [showRouteNumbers, setShowRouteNumbers] = useState(false);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedRouteId, setSelectedRouteId] = useState('');
  const [selectedStopNodeId, setSelectedStopNodeId] = useState('');
  const [showAllRoutes, setShowAllRoutes] = useState(false);

  useEffect(() => {
    const storedDataset = localStorage.getItem(DATASET_STORAGE_KEY);
    if (storedDataset) {
      const parsed = JSON.parse(storedDataset);
      setDataset(parsed);
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
        // keep silent for now
      }
    };

    loadMeta();
  }, [dataset]);

  const selectedRoute: Route | null = useMemo(() => {
    if (!run || !selectedRouteId) return null;
    return run.routes.find((route) => route.id === selectedRouteId) ?? null;
  }, [run, selectedRouteId]);

  const selectedRoutesForDisplay = useMemo(() => {
    if (!run) return [];
    if (showAllRoutes) return run.routes;
    if (!selectedRouteId) return run.routes;
    const match = run.routes.find((route) => route.id === selectedRouteId);
    return match ? [match] : run.routes;
  }, [run, selectedRouteId, showAllRoutes]);

  const displayDepot: Depot | null = useMemo(() => {
    if (run?.previewSummary?.depotLat != null && run?.previewSummary?.depotLon != null) {
      return {
        id: 'PREVIEW-DEPOT',
        name: 'Depot',
        lat: run.previewSummary.depotLat,
        lon: run.previewSummary.depotLon,
      };
    }
    return depot;
  }, [run, depot]);

  const selectedStopOptions = useMemo(() => {
    if (!selectedRoute) return [];
    return selectedRoute.stops.map((stop) => ({
      value: stop.nodeId,
      label: `Stop ${stop.stopNumber} - ${stop.nodeName}`,
    }));
  }, [selectedRoute]);

  useEffect(() => {
    if (selectedRoute?.stops?.length) {
      setSelectedStopNodeId(selectedRoute.stops[0].nodeId);
    } else {
      setSelectedStopNodeId('');
    }
  }, [selectedRoute]);

  const handleRun = async () => {
    if (!dataset) {
      setMessage('No validated dataset found. Please upload and validate first.');
      return;
    }

    try {
      setBusy(true);
      setMessage('Running baseline experiment on backend...');

      const requestedReps = Number(parameters.numRepresentatives) || 4;
      const clampedReps = Math.max(2, Math.min(15, requestedReps));

      setAddedCustomers([]);

      const result = await runBaseline({
        datasetId: dataset.id,
        numRepresentatives: clampedReps,
        avgSpeedKmph: Number(parameters.speed) || 18.75,
        serviceMinutesPerStop: Number(parameters.serviceMinutes) || 8,
        seed: Number(parameters.seed) || 42,
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

      localStorage.setItem(BASELINE_STORAGE_KEY, JSON.stringify(summary));
      setRun(result);
      setSelectedRouteId(result.routes[0]?.id ?? '');
      setMessage('Baseline run completed.');
      setMapLoaded(false);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : 'Baseline run failed.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="h-screen flex flex-col">
        <div className="bg-white border-b border-gray-200 px-8 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Baseline Route Generation</h1>
            <p className="text-sm text-gray-600">
              Greedy Nearest Neighbor + Dijkstra Shortest Path
            </p>
          </div>

          <Button variant="outline" onClick={() => navigate('/')}>
            <span className="inline-flex items-center gap-2">
              <ArrowLeftIcon className="w-4 h-4" />
              Back to Upload
            </span>
          </Button>
        </div>

        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 p-6 overflow-hidden min-w-0 flex flex-col">
            {run ? (
              <>
                <div className="flex-1 min-h-[480px]">
                  <MapCanvas
                    routes={selectedRoutesForDisplay}
                    depot={displayDepot}
                    showLabels={showLabels}
                    showRouteNumbers={showRouteNumbers}
                    highlightedNodes={selectedStopNodeId ? [selectedStopNodeId] : []}
                    onMapReady={() => setMapLoaded(true)}
                    addedCustomers={addedCustomers}
                  />
                </div>

                <div className="mt-6">
                  <RouteTable
                    routes={selectedRoute ? [selectedRoute] : []}
                    title={
                      selectedRoute
                        ? `Route: ${formatSalesRepName(selectedRoute.representativeName)}`
                        : 'Selected Route'
                    }
                  />
                </div>
              </>
            ) : (
              <Card className="h-full flex items-center justify-center">
                <div className="text-center">
                  <MapIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">Run the baseline algorithm to visualize routes</p>
                </div>
              </Card>
            )}
          </div>

          <div className="w-[420px] bg-white border-l border-gray-200 p-6 overflow-auto">
            <Card className="mb-6">
              <h2 className="text-sm font-semibold text-gray-900 mb-4">Parameters</h2>

              <div className="space-y-3">
                <div className="rounded border border-gray-200 bg-gray-50 px-3 py-2">
                  <div className="text-xs text-gray-500">Depot</div>
                  <div className="text-sm font-medium text-gray-900">
                    {dataset?.datasetRole === 'primary_reconstruction'
                      ? 'DEPOT-130'
                      : dataset?.datasetRole === 'comparative_template'
                        ? 'DEPOT-153'
                        : 'Automatic'}
                  </div>
                </div>

                <Input
                  label="Average Speed (km/h)"
                  type="number"
                  value={parameters.speed}
                  onChange={(val) => setParameters((prev) => ({ ...prev, speed: val }))}
                />

                <Input
                  label="Service Minutes per Stop"
                  type="number"
                  value={parameters.serviceMinutes}
                  onChange={(val) => setParameters((prev) => ({ ...prev, serviceMinutes: val }))}
                />

                {/* Random seed is kept internal for reproducibility and hidden from the UI */}
              </div>

              <Button
                onClick={handleRun}
                disabled={busy}
                className="w-full mt-4 flex items-center justify-center gap-2"
              >
                {busy ? (
                  <span>Processing...</span>
                ) : (
                  <>
                    <PlayIcon className="w-4 h-4" />
                    <span>Run Baseline Algorithm</span>
                  </>
                )}
              </Button>

              {mapLoaded && (
                <>
                  <Button
                    variant="outline"
                    onClick={() => setShowAddModal(true)}
                    className="w-full mt-3 flex items-center justify-center gap-2"
                  >
                    <PlusIcon className="w-4 h-4" />
                    <span>Add Customer</span>
                  </Button>

                  <AddCustomerModal
                    isOpen={showAddModal}
                    onClose={() => setShowAddModal(false)}
                    depot={displayDepot}
                    onConfirm={async (customers: AddCustomer[]) => {
                      if (!run) {
                        setMessage('Run the baseline first before adding customers.');
                        setShowAddModal(false);
                        return;
                      }

                      const currentRun = run;
                      const existingAdded = addedCustomers;

                      let nextNumber = getNextCustomerNumber(currentRun, existingAdded);

                      const normalized: AddedCustomer[] = customers
                        .filter((c) => c.lat != null && c.lon != null)
                        .map((c, idx) => {
                          const lat = Number(c.lat);
                          const lon = Number(c.lon);
                          const customerNumber = nextNumber++;

                          return {
                            id: `ADDED-${Date.now()}-${idx}`,
                            customerNumber,
                            // Let the backend assign this to the nearest representative route.
                            assignedRep: '',
                            label: `Customer ${customerNumber}`,
                            lat,
                            lon,
                            address: c.address,
                          };
                        });

                      // Prepend new pending customers so the newest appear first in the preview.
                      const pendingPreview = [...normalized, ...existingAdded];
                      setAddedCustomers(pendingPreview);

                      try {
                        setBusy(true);
                        setMessage(
                          `Adding ${normalized.length} customer(s) and rerouting baseline...`
                        );

                        const updatedRun = await addCustomersToBaseline(currentRun.id, normalized);

                        const summary: StoredRunSummary = {
                          id: updatedRun.id,
                          algorithm: updatedRun.algorithm,
                          datasetId: updatedRun.datasetId,
                          kpis: updatedRun.kpis,
                          representatives: updatedRun.representatives,
                        };

                        localStorage.setItem(BASELINE_STORAGE_KEY, JSON.stringify(summary));
                        // Keep the old added customers and append the newly processed customers
                        // with their backend-assigned representatives.
                        const assignedPreview = applyBackendAddedCustomerAssignments(
                          updatedRun,
                          normalized
                        );
                        setAddedCustomers([...assignedPreview, ...existingAdded]);
                        setRun(updatedRun);
                        setSelectedRouteId(updatedRun.routes[0]?.id ?? '');
                        setMessage(
                          `Added ${normalized.length} customer(s) and updated baseline routes.`
                        );
                        setShowAddModal(false);
                        setMapLoaded(false);
                      } catch (err) {
                        setMessage(
                          err instanceof Error
                            ? err.message
                            : 'Failed to add customers to baseline.'
                        );
                      } finally {
                        setBusy(false);
                      }
                    }}
                  />
                  {addedCustomers.length > 0 && (
                    <Card className="mt-3">
                      <h3 className="text-sm font-semibold text-gray-900 mb-2">
                        Added Customers Preview
                      </h3>
                      <div className="space-y-2 text-sm text-gray-700 max-h-96 overflow-auto pr-2">
                        {addedCustomers.map((customer) => (
                          <div key={customer.id} className="rounded border border-gray-200 p-2">
                            <div className="font-medium">{customer.label}</div>
                            <div>
                              Rep:{' '}
                              {customer.assignedRep ? (
                                <span className="font-medium text-green-700">
                                  {customer.assignedRep}
                                </span>
                              ) : (
                                <span className="text-gray-500">assigned after processing</span>
                              )}
                            </div>
                            <div>Lat: {customer.lat.toFixed(6)}</div>
                            <div>Lon: {customer.lon.toFixed(6)}</div>
                            {customer.address && (
                              <div className="text-xs text-gray-500 mt-1">{customer.address}</div>
                            )}
                          </div>
                        ))}
                      </div>
                    </Card>
                  )}
                </>
              )}

              {message && <div className="mt-3 text-sm text-slate-700">{message}</div>}
            </Card>

            {run && (
              <>
                <div className="mb-6">
                  <h2 className="text-sm font-semibold text-gray-900 mb-3">
                    Key Performance Indicators
                  </h2>

                  <div className="space-y-3">
                    <KPICard
                      title="Total Distance"
                      value={run.kpis.totalDistance}
                      unit="km"
                      icon={<RouteIcon className="w-5 h-5" />}
                    />
                    <KPICard
                      title="Travel Time"
                      value={run.kpis.travelTime}
                      unit="hr"
                      icon={<ClockIcon className="w-5 h-5" />}
                    />
                    <KPICard
                      title="Operational Time"
                      value={run.kpis.operationalTime}
                      unit="hr"
                      icon={<ClockIcon className="w-5 h-5" />}
                    />
                    <KPICard
                      title="Fairness"
                      value={run.kpis.fairness}
                      icon={<UsersIcon className="w-5 h-5" />}
                    />
                    <KPICard
                      title="CPU Runtime"
                      value={
                        run?.cpuRuntimeSeconds !== undefined
                          ? run.cpuRuntimeSeconds.toFixed(3)
                          : "—"
                      }
                      unit="sec"
                      icon={<ClockIcon className="w-5 h-5" />}
                    />
                  </div>
                </div>

                <div className="mb-6">
                  <h2 className="text-sm font-semibold text-gray-900 mb-3">Map Controls</h2>

                  <div className="space-y-2">
                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={showLabels}
                        onChange={(e) => setShowLabels(e.target.checked)}
                      />
                      Show Customer Labels and assigned reps
                    </label>

                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={showRouteNumbers}
                        onChange={(e) => setShowRouteNumbers(e.target.checked)}
                      />
                      Show Route Numbers
                    </label>

                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={showAllRoutes}
                        onChange={(e) => setShowAllRoutes(e.target.checked)}
                      />
                      Show All Representatives
                    </label>
                  </div>
                </div>

                {addedCustomers.length > 0 && (
                  <Card className="mt-6">
                    <h2 className="text-sm font-semibold text-gray-900 mb-3">
                      Added Customers Preview
                    </h2>
                    <div className="space-y-2 text-sm text-gray-700">
                      {addedCustomers.map((customer) => (
                        <div key={customer.id} className="rounded border border-gray-200 p-2">
                          <div className="font-medium">{customer.label}</div>
                          {customer.assignedRep && (
                            <div>Rep: {formatSalesRepName(customer.assignedRep)}</div>
                          )}
                          <div>Lat: {customer.lat.toFixed(6)}</div>
                          <div>Lon: {customer.lon.toFixed(6)}</div>
                          {customer.address && (
                            <div className="text-xs text-gray-500 mt-1">{customer.address}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </Card>
                )}

                <div className="mb-6">
                  <Select
                    label="Select Route"
                    value={selectedRouteId}
                    onChange={setSelectedRouteId}
                    options={run.routes.map((route) => ({
                      value: route.id,
                      label: formatSalesRepName(route.representativeName),
                    }))}
                  />
                </div>

                <div className="mb-6">
                  <Select
                    label="Select Stop"
                    value={selectedStopNodeId}
                    onChange={setSelectedStopNodeId}
                    options={selectedStopOptions}
                  />
                </div>

                <Button
                  onClick={() => navigate('/enhanced')}
                  variant="secondary"
                  className="w-full mt-6"
                  disabled={!run}
                >
                  Use Baseline as Seed for Enhanced Model
                </Button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BaselineRun;
