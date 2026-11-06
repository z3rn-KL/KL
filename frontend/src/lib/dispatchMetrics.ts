import type { DispatchOrderItem, ShipperData } from "@/lib/api";

/**
 * Shared presentation selectors. These never mutate session state or recalculate
 * routing decisions. Counts use exclusive order IDs and explicit day semantics.
 */
export function newOrdersOnDay(orders: DispatchOrderItem[], day: number): number {
  return orders.filter((order) => order.release_day === day).length;
}

export function carryoverEnteringDay(orders: DispatchOrderItem[], day: number): number {
  if (day <= 1) return 0;
  return orders.filter((order) =>
    order.release_day < day &&
    (order.delivered_day == null || order.delivered_day >= day)
  ).length;
}

export function deliveredOnDay(orders: DispatchOrderItem[], day: number): number {
  return orders.filter((order) =>
    order.delivery_status === "DELIVERED" && order.delivered_day === day
  ).length;
}

/** A snapshot may be taken after later days have completed.
 * The delivered_day is the completion date; only deliveries finished before
 * the requested day are excluded from that day's opening carryover.
 */
export function workloadForDay(orders: DispatchOrderItem[], day: number): number {
  return newOrdersOnDay(orders, day) + carryoverEnteringDay(orders, day);
}

/** IDs outstanding at the close of an historical day. Reading the current
 * delivery_status instead would lose those later delivered on a future day.
 */
export function outstandingAtEndOfDay(orders: DispatchOrderItem[], day: number): DispatchOrderItem[] {
  return orders.filter((order) =>
    order.release_day <= day && (order.delivered_day == null || order.delivered_day > day)
  );
}

export function pendingAssignedWeightKg(orders: DispatchOrderItem[]): number {
  return orders.filter((order) =>
    order.delivery_status === "ASSIGNED" || order.delivery_status === "IN_TRANSIT"
  ).reduce((sum, order) => sum + (order.weight_kg || 0), 0);
}

export function ordersBelongingToShipper(orders: DispatchOrderItem[], shipper: ShipperData | undefined): DispatchOrderItem[] {
  if (!shipper) return [];
  const ids = new Set([...shipper.assigned_order_ids, ...shipper.completed_order_ids]);
  return orders.filter((order) => ids.has(order.order_id) || order.assigned_shipper_id === shipper.shipper_id);
}

export function deliveredSlaCounts(orders: DispatchOrderItem[]) {
  const delivered = orders.filter((order) => order.delivery_status === "DELIVERED");
  const onTime = delivered.filter((order) => order.sla_status === "ON_TIME").length;
  const overdue = delivered.filter((order) => order.sla_status === "OVERDUE").length;
  return {
    delivered: delivered.length,
    onTime,
    overdue,
    unknown: delivered.length - onTime - overdue,
    ratePct: delivered.length ? Math.round(onTime / delivered.length * 100) : null,
  };
}

/** Shippers who received or completed eligible work in the specified day.
 * Unlike global shipper.completed_order_ids this never counts Day 1 as Day 2.
 */
export function activeShippersOnDay(orders: DispatchOrderItem[], shippers: ShipperData[], day: number): number {
  const relevant = orders.filter((order) =>
    order.release_day === day || order.delivered_day === day ||
    (order.release_day < day && (order.delivered_day == null || order.delivered_day >= day))
  );
  return shippers.filter((shipper) => {
    const knownOrders = new Set([...shipper.assigned_order_ids, ...shipper.completed_order_ids]);
    return relevant.some((order) => order.assigned_shipper_id === shipper.shipper_id || knownOrders.has(order.order_id));
  }).length;
}
