import { useState, useEffect, useRef, useMemo } from "react";
import { ShipperData } from "@/lib/api";

export interface ShipperPose {
  shipperId: string;
  lat: number;
  lng: number;
  heading: number;
  status: "IDLE" | "AT_BASE" | "MOVING_TO_DEPOT" | "PICKING_UP" | "IN_TRANSIT" | "DELIVERING" | "RETURNING" | "COMPLETED";
  statusText: string;
  currentStopIndex: number;
  totalStops: number;
  currentStopName?: string;
  nextStopName?: string;
  legProgress: number; // 0..1
}

interface Waypoint {
  lat: number;
  lng: number;
  arrMin: number;
  depMin: number;
  type: "START" | "DEPOT_PICKUP" | "STOP" | "RETURN_DEPOT";
  name: string;
  orderId?: string;
}

function toRad(deg: number): number {
  return (deg * Math.PI) / 180;
}

function toDeg(rad: number): number {
  return (rad * 180) / Math.PI;
}

/**
 * Calculates compass bearing (heading angle in degrees) from point 1 to point 2.
 */
export function calculateBearing(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const dLng = toRad(lng2 - lng1);
  const rLat1 = toRad(lat1);
  const rLat2 = toRad(lat2);

  const y = Math.sin(dLng) * Math.cos(rLat2);
  const x = Math.cos(rLat1) * Math.sin(rLat2) - Math.sin(rLat1) * Math.cos(rLat2) * Math.cos(dLng);

  let brng = toDeg(Math.atan2(y, x));
  return (brng + 360) % 360;
}

interface UseShipperVehicleAnimationProps {
  shippers: ShipperData[];
  simulationClock: number; // Backend simulation minutes (0, 15, 30...)
  isPlaying: boolean;
  simSpeed: number; // 1, 5, 10, 20
  depot: { lat: number; lng: number; name?: string };
  selectedShipperId: string;
}

export function useShipperVehicleAnimation({
  shippers,
  simulationClock,
  isPlaying,
  simSpeed,
  depot,
  selectedShipperId,
}: UseShipperVehicleAnimationProps) {
  // Smooth presentation clock in minutes
  const [presentationClock, setPresentationClock] = useState<number>(simulationClock);
  const [autoFollow, setAutoFollow] = useState<boolean>(false);

  // References for animation loop
  const clockRef = useRef<number>(simulationClock);
  const targetClockRef = useRef<number>(simulationClock);
  const lastTimeRef = useRef<number>(performance.now());
  const rafRef = useRef<number | null>(null);

  // When backend clock updates (e.g. from step or polling tick)
  useEffect(() => {
    targetClockRef.current = simulationClock;
    if (!isPlaying) {
      clockRef.current = simulationClock;
      setPresentationClock(simulationClock);
    }
  }, [simulationClock, isPlaying]);

  // Animation frame loop for continuous smooth motion
  useEffect(() => {
    lastTimeRef.current = performance.now();

    const loop = (now: number) => {
      const deltaSec = (now - lastTimeRef.current) / 1000;
      lastTimeRef.current = now;

      if (isPlaying) {
        // Simulation speed: 15 sim minutes per (1500ms / simSpeed) = 10 * simSpeed sim minutes per real second
        const simRate = 10 * simSpeed;
        const newClock = clockRef.current + deltaSec * simRate;

        // Keep presentation clock advancing smoothly up to slightly ahead of backend tick
        clockRef.current = newClock;
        setPresentationClock(newClock);
      } else {
        // When paused or stepping, smoothly slide to targetClockRef if there's a difference
        const diff = targetClockRef.current - clockRef.current;
        if (Math.abs(diff) > 0.05) {
          const step = diff * Math.min(1, deltaSec * 8); // smooth catch-up
          clockRef.current += step;
          setPresentationClock(clockRef.current);
        } else {
          clockRef.current = targetClockRef.current;
        }
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [isPlaying, simSpeed]);

  // Compute poses for all shippers at presentationClock
  const poses = useMemo(() => {
    const result: Record<string, ShipperPose> = {};
    const t = presentationClock;

    for (const s of shippers) {
      const isModeB = s.shift_start_mode === "START_AT_BASE";
      const startLat = isModeB && s.base_lat ? s.base_lat : (s.start_lat || depot.lat);
      const startLng = isModeB && s.base_lng ? s.base_lng : (s.start_lng || depot.lng);

      if (!s.timeline_stops || s.timeline_stops.length === 0) {
        result[s.shipper_id] = {
          shipperId: s.shipper_id,
          lat: startLat,
          lng: startLng,
          heading: 0,
          status: isModeB ? "AT_BASE" : "IDLE",
          statusText: isModeB ? `Tại trạm: ${s.base_name || "Vệ tinh"}` : "Sẵn sàng tại Kho chính",
          currentStopIndex: 0,
          totalStops: 0,
          legProgress: 0,
        };
        continue;
      }

      // Build route waypoints
      const waypoints: Waypoint[] = [
        {
          lat: startLat,
          lng: startLng,
          arrMin: 0,
          depMin: 0,
          type: "START",
          name: isModeB ? (s.base_name || "Trạm vệ tinh") : (depot.name || "Kho trung tâm"),
        },
      ];

      for (const st of s.timeline_stops) {
        waypoints.push({
          lat: st.lat,
          lng: st.lng,
          arrMin: st.arrival_minute,
          depMin: st.departure_minute,
          type: ((st as any).stop_type === "DEPOT_PICKUP" || (st as any).is_depot_pickup || st.order_id === "depot_pickup" || st.order_id === "DEPOT_PICKUP") ? "DEPOT_PICKUP" : "STOP",
          name: st.package_name || st.order_code || st.order_id,
          orderId: st.order_id,
        });
      }

      // Final return leg to Central Depot
      const lastWp = waypoints[waypoints.length - 1];
      const returnDist = Math.hypot(depot.lat - lastWp.lat, depot.lng - lastWp.lng) * 111.0;
      const returnMin = (returnDist / 25.0) * 60.0;
      const returnArrMin = lastWp.depMin + Math.max(1.0, returnMin);

      waypoints.push({
        lat: depot.lat,
        lng: depot.lng,
        arrMin: returnArrMin,
        depMin: returnArrMin + 60,
        type: "RETURN_DEPOT",
        name: `Về ${depot.name || "Kho trung tâm"}`,
      });

      // Find current position between waypoints
      let currLat = startLat;
      let currLng = startLng;
      let currHeading = 0;
      let status: ShipperPose["status"] = "IDLE";
      let statusText = "Khởi động ca";
      let currentStopIndex = 0;
      let currentStopName = "";
      let nextStopName = "";
      let legProgress = 0;

      if (t <= 0) {
        currLat = startLat;
        currLng = startLng;
        status = isModeB ? "AT_BASE" : "IDLE";
        statusText = isModeB ? `Tại ${s.base_name || "Trạm trực ca"}` : "Tại Kho trung tâm";
        nextStopName = waypoints[1]?.name || "";
      } else {
        let found = false;

        for (let i = 0; i < waypoints.length - 1; i++) {
          const wCurrent = waypoints[i];
          const wNext = waypoints[i + 1];

          // 1. Staying at wCurrent (loading/delivering)
          if (t >= wCurrent.arrMin && t <= wCurrent.depMin && i > 0) {
            currLat = wCurrent.lat;
            currLng = wCurrent.lng;
            currHeading = calculateBearing(waypoints[i - 1].lat, waypoints[i - 1].lng, wCurrent.lat, wCurrent.lng);
            currentStopIndex = i;
            currentStopName = wCurrent.name;
            nextStopName = wNext.name;
            legProgress = 1.0;

            if (wCurrent.type === "DEPOT_PICKUP") {
              status = "PICKING_UP";
              statusText = "Đang lấy hàng tại Kho trung tâm";
            } else {
              status = "DELIVERING";
              statusText = `Đang giao đơn: ${wCurrent.name}`;
            }
            found = true;
            break;
          }

          // 2. Traveling between wCurrent and wNext
          if (t > wCurrent.depMin && t < wNext.arrMin) {
            const legDuration = Math.max(0.1, wNext.arrMin - wCurrent.depMin);
            legProgress = Math.max(0, Math.min(1, (t - wCurrent.depMin) / legDuration));

            currLat = wCurrent.lat + legProgress * (wNext.lat - wCurrent.lat);
            currLng = wCurrent.lng + legProgress * (wNext.lng - wCurrent.lng);
            currHeading = calculateBearing(wCurrent.lat, wCurrent.lng, wNext.lat, wNext.lng);
            currentStopIndex = i + 1;
            nextStopName = wNext.name;

            if (wNext.type === "DEPOT_PICKUP") {
              status = "MOVING_TO_DEPOT";
              statusText = "Di chuyển đến Kho trung tâm nhận kiện";
            } else if (wNext.type === "RETURN_DEPOT") {
              status = "RETURNING";
              statusText = "Đang trở về Kho trung tâm";
            } else {
              status = "IN_TRANSIT";
              statusText = `Đang di chuyển đến: ${wNext.name}`;
            }
            found = true;
            break;
          }
        }

        if (!found) {
          // Beyond all waypoints: completed shift at depot
          currLat = depot.lat;
          currLng = depot.lng;
          currHeading = 0;
          status = "COMPLETED";
          statusText = "Đã hoàn thành ca giao và về Kho";
          currentStopIndex = s.timeline_stops.length;
          legProgress = 1.0;
        }
      }

      result[s.shipper_id] = {
        shipperId: s.shipper_id,
        lat: currLat,
        lng: currLng,
        heading: currHeading,
        status,
        statusText,
        currentStopIndex,
        totalStops: s.timeline_stops.length,
        currentStopName,
        nextStopName,
        legProgress,
      };
    }

    return result;
  }, [shippers, presentationClock, depot]);

  // Selected shipper pose
  const selectedShipperPose = poses[selectedShipperId] || poses[shippers[0]?.shipper_id];

  // Format digital clock (starts at 08:00)
  const pad = (n: number | string) => String(n).padStart(2, "0");
  const startHour = 8;
  const totalMins = startHour * 60 + Math.floor(presentationClock);
  const hours = Math.floor(totalMins / 60) % 24;
  const minutes = Math.floor(totalMins % 60);
  const displayClock = `${pad(hours)}:${pad(minutes)}`;

  return {
    presentationClock,
    displayClock,
    poses,
    selectedShipperPose,
    autoFollow,
    setAutoFollow,
  };
}
