import React from 'react';
import type { Route } from '../types';
import { Card } from './Card';

interface RouteTableProps {
  routes: Route[];
  title?: string;
}

const formatSalesRepName = (repId?: string | null) => {
  if (!repId) return '';
  return repId.replace('-AGE-', '-');
};

export function RouteTable({ routes, title }: RouteTableProps) {
  const allStops = routes.flatMap((route) => {
  const normalStops = route.stops.map((stop) => ({
    ...stop,
    routeId: route.id,
    representativeName: route.representativeName,
    routeColor: route.color,
    isReturnToDepot: false,
  }));

  const lastStop = route.stops[route.stops.length - 1];

  if (!lastStop) {
    return normalStops;
  }

  const returnLegDistance = route.returnLegDistance ?? 0;
  const routeDistance =
    route.routeDistance ??
    ((lastStop.cumulativeDistance ?? 0) + returnLegDistance);

  const returnRow = {
    stopNumber: 'Return',
    nodeId: 'DEPOT',
    nodeName: 'Depot',
    orderCount: 0,
    lat: 0,
    lon: 0,
    orderId: '-',
    predictedEtaMin: 0,
    legDistance: returnLegDistance,
    cumulativeDistance: routeDistance,
    eta: lastStop.eta,
    routeId: route.id,
    representativeName: route.representativeName,
    routeColor: route.color,
    isReturnToDepot: true,
  };

  return [...normalStops, returnRow];
});

  return (
    <Card padding="none">
      {title && (
        <div className="px-4 py-3 border-b border-gray-200">
          <h3 className="text-sm font-semibold text-gray-900">{title}</h3>
        </div>
      )}

      <div className="overflow-auto h-56 lg:h-56">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <th className="px-4 py-2 text-left font-medium text-gray-700">Rep</th>
              <th className="px-4 py-2 text-left font-medium text-gray-700">Stop</th>
              <th className="px-4 py-2 text-left font-medium text-gray-700">Customer</th>
              <th className="px-4 py-2 text-left font-medium text-gray-700">Order ID</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700">Leg (km)</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700">Cumulative (km)</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700">ETA Proxy (min)</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-gray-200">
            {allStops.map((stop) => (
              <tr
                key={`${stop.routeId}-${stop.stopNumber}-${stop.nodeId}`}
                className="hover:bg-gray-50"
              >
                <td className="px-4 py-2 text-gray-900">
                  {formatSalesRepName(stop.representativeName)}
                </td>
                <td className="px-4 py-2 text-gray-900">{stop.stopNumber}</td>
                <td className="px-4 py-2 text-gray-900">{stop.nodeName}</td>
                <td className="px-4 py-2 text-gray-600">{stop.orderId ?? '-'}</td>
                <td className="px-4 py-2 text-right text-gray-600">
                  {stop.legDistance.toFixed(2)}
                </td>
                <td className="px-4 py-2 text-right text-gray-600">
                  {stop.cumulativeDistance.toFixed(2)}
                </td>
                <td className="px-4 py-2 text-right text-gray-600">{stop.eta.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
