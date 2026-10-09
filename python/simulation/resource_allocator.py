from dataclasses import dataclass

from domain import (
    Assignment,
    Cluster,
    Delivery,
    Shipper,
    Vehicle,
)


@dataclass
class AllocationWorkload:
    """
    Workload assigned to one vehicle and
    optionally one shipper.

    One vehicle is kept inside one spatial
    cluster during a planning wave.
    """

    workload_id: str

    cluster_id: int

    depot_id: str | None

    vehicle_id: str

    shipper_id: str | None

    deliveries: list[Delivery]

    total_weight_kg: float

    capacity_kg: float

    max_orders: int | None

    @property
    def number_of_deliveries(
        self,
    ) -> int:
        return len(
            self.deliveries
        )

    @property
    def remaining_capacity_kg(
        self,
    ) -> float:
        return (
            self.capacity_kg
            - self.total_weight_kg
        )

    @property
    def capacity_utilization(
        self,
    ) -> float:
        if self.capacity_kg <= 0.0:
            return 0.0

        return (
            self.total_weight_kg
            / self.capacity_kg
        )


@dataclass
class ResourceAllocationResult:
    """
    Result of cluster-to-resource
    allocation.
    """

    assignments: list[Assignment]

    workloads: list[AllocationWorkload]

    unassigned_delivery_ids: list[str]

    total_deliveries: int

    assigned_deliveries: int

    unassigned_deliveries: int

    used_vehicles: int

    used_shippers: int

    feasible: bool


@dataclass
class _ResourceSlot:
    """
    Internal mutable resource state.
    """

    vehicle: Vehicle

    shipper: Shipper | None

    cluster_id: int | None = None

    deliveries: list[Delivery] | None = None

    total_weight_kg: float = 0.0

    def __post_init__(
        self,
    ) -> None:
        if self.deliveries is None:
            self.deliveries = []

    @property
    def remaining_capacity_kg(
        self,
    ) -> float:
        return (
            float(
                self.vehicle.capacity_kg
            )
            - self.total_weight_kg
        )

    @property
    def remaining_orders(
        self,
    ) -> int | None:
        if self.shipper is None:
            return None

        if self.shipper.max_orders is None:
            return None

        return (
            int(
                self.shipper.max_orders
            )
            - len(
                self.deliveries
            )
        )

    def can_accept(
        self,
        delivery: Delivery,
        cluster_id: int,
        weight_kg: float,
    ) -> bool:
        # A vehicle must remain inside one
        # spatial cluster for the wave.
        if (
            self.cluster_id is not None
            and self.cluster_id
            != cluster_id
        ):
            return False

        if (
            weight_kg
            > self.remaining_capacity_kg
            + 1e-9
        ):
            return False

        remaining_orders = (
            self.remaining_orders
        )

        if (
            remaining_orders is not None
            and remaining_orders <= 0
        ):
            return False

        return True

    def add(
        self,
        delivery: Delivery,
        cluster_id: int,
        weight_kg: float,
    ) -> None:
        if self.cluster_id is None:
            self.cluster_id = (
                cluster_id
            )

        self.deliveries.append(
            delivery
        )

        self.total_weight_kg += (
            weight_kg
        )


class ResourceAllocator:
    """
    Allocate clustered deliveries to
    vehicle/shipper resources.

    Current operational constraints:

    1. vehicle capacity;
    2. shipper max_orders;
    3. depot compatibility;
    4. one vehicle serves only one spatial
       cluster during a planning wave.

    max_work_hours is intentionally not
    enforced here because route duration is
    not known until routing is completed.
    It will be evaluated after routing.
    """

    def __init__(
        self,
        vehicles: list[Vehicle],
        shippers: list[Shipper],
        require_shipper: bool = True,
    ) -> None:
        if not vehicles:
            raise ValueError(
                "Danh sách vehicle rỗng."
            )

        if (
            require_shipper
            and not shippers
        ):
            raise ValueError(
                "require_shipper=True nhưng "
                "không có shipper."
            )

        self.vehicles = (
            list(
                vehicles
            )
        )

        self.shippers = (
            list(
                shippers
            )
        )

        self.require_shipper = (
            require_shipper
        )

        self._validate_resources()

    def allocate(
        self,
        clusters: list[Cluster],
    ) -> ResourceAllocationResult:
        """
        Allocate all deliveries contained
        in the supplied clusters.

        Strategy:

        - create compatible vehicle/shipper
          resource slots;
        - process larger clusters first;
        - process heavier deliveries first;
        - prefer an already-open workload
          from the same cluster;
        - otherwise open a free vehicle;
        - use best-fit remaining capacity.

        This is a deterministic greedy
        resource-allocation heuristic.
        """

        if not clusters:
            raise ValueError(
                "Danh sách cluster rỗng."
            )

        slots = (
            self._build_resource_slots()
        )

        if not slots:
            raise ValueError(
                "Không có cặp resource khả dụng."
            )

        assignments = []

        unassigned = []

        total_deliveries = sum(
            len(
                cluster.deliveries
            )
            for cluster
            in clusters
        )

        ordered_clusters = sorted(
            clusters,
            key=lambda cluster: (
                -self._cluster_total_weight(
                    cluster
                ),
                cluster.cluster_id,
            ),
        )

        for cluster in (
            ordered_clusters
        ):
            deliveries = sorted(
                cluster.deliveries,
                key=lambda delivery: (
                    -self._delivery_weight_kg(
                        delivery
                    ),
                    str(
                        delivery.delivery_id
                    ),
                ),
            )

            for delivery in deliveries:
                weight_kg = (
                    self._delivery_weight_kg(
                        delivery
                    )
                )

                slot = (
                    self._select_slot(
                        slots=slots,
                        cluster_id=(
                            cluster.cluster_id
                        ),
                        delivery=delivery,
                        weight_kg=weight_kg,
                    )
                )

                if slot is None:
                    unassigned.append(
                        str(
                            delivery.delivery_id
                        )
                    )

                    continue

                slot.add(
                    delivery=delivery,
                    cluster_id=(
                        cluster.cluster_id
                    ),
                    weight_kg=weight_kg,
                )

                assignments.append(
                    Assignment(
                        delivery_id=str(
                            delivery.delivery_id
                        ),
                        cluster_id=(
                            cluster.cluster_id
                        ),
                        depot_id=(
                            slot.vehicle.depot_id
                        ),
                        vehicle_id=(
                            slot.vehicle.vehicle_id
                        ),
                        shipper_id=(
                            None
                            if slot.shipper
                            is None
                            else slot.shipper
                            .shipper_id
                        ),
                    )
                )

        workloads = (
            self._build_workloads(
                slots
            )
        )

        used_vehicle_ids = {
            workload.vehicle_id
            for workload
            in workloads
        }

        used_shipper_ids = {
            workload.shipper_id
            for workload
            in workloads
            if workload.shipper_id
            is not None
        }

        assigned_count = len(
            assignments
        )

        return ResourceAllocationResult(
            assignments=assignments,
            workloads=workloads,
            unassigned_delivery_ids=(
                unassigned
            ),
            total_deliveries=(
                total_deliveries
            ),
            assigned_deliveries=(
                assigned_count
            ),
            unassigned_deliveries=(
                len(
                    unassigned
                )
            ),
            used_vehicles=len(
                used_vehicle_ids
            ),
            used_shippers=len(
                used_shipper_ids
            ),
            feasible=(
                len(
                    unassigned
                )
                == 0
            ),
        )

    def _build_resource_slots(
        self,
    ) -> list[_ResourceSlot]:
        vehicles = sorted(
            self.vehicles,
            key=lambda vehicle: (
                vehicle.vehicle_id
            ),
        )

        unused_shippers = sorted(
            self.shippers,
            key=lambda shipper: (
                shipper.shipper_id
            ),
        )

        slots = []

        for vehicle in vehicles:
            shipper = (
                self._take_compatible_shipper(
                    vehicle=vehicle,
                    unused_shippers=(
                        unused_shippers
                    ),
                )
            )

            if (
                self.require_shipper
                and shipper is None
            ):
                continue

            slots.append(
                _ResourceSlot(
                    vehicle=vehicle,
                    shipper=shipper,
                )
            )

        return slots

    def _take_compatible_shipper(
        self,
        vehicle: Vehicle,
        unused_shippers: list[Shipper],
    ) -> Shipper | None:
        for index, shipper in enumerate(
            unused_shippers
        ):
            if (
                shipper.depot_id is None
                or shipper.depot_id
                == vehicle.depot_id
            ):
                return unused_shippers.pop(
                    index
                )

        return None

    def _select_slot(
        self,
        slots: list[_ResourceSlot],
        cluster_id: int,
        delivery: Delivery,
        weight_kg: float,
    ) -> _ResourceSlot | None:
        candidates = [
            slot
            for slot
            in slots
            if slot.can_accept(
                delivery=delivery,
                cluster_id=cluster_id,
                weight_kg=weight_kg,
            )
        ]

        if not candidates:
            return None

        # Prefer slots already assigned to
        # the same cluster. This avoids
        # unnecessarily consuming vehicles.
        same_cluster = [
            slot
            for slot
            in candidates
            if slot.cluster_id
            == cluster_id
        ]

        if same_cluster:
            candidates = (
                same_cluster
            )

        else:
            candidates = [
                slot
                for slot
                in candidates
                if slot.cluster_id
                is None
            ]

        if not candidates:
            return None

        # Best-fit capacity:
        # select the feasible vehicle with
        # the smallest remaining capacity
        # after inserting this delivery.
        return min(
            candidates,
            key=lambda slot: (
                slot.remaining_capacity_kg
                - weight_kg,
                slot.vehicle.vehicle_id,
            ),
        )

    def _build_workloads(
        self,
        slots: list[_ResourceSlot],
    ) -> list[AllocationWorkload]:
        workloads = []

        for slot in slots:
            if not slot.deliveries:
                continue

            cluster_id = int(
                slot.cluster_id
            )

            workload_id = (
                f"cluster_{cluster_id}"
                f"__vehicle_"
                f"{slot.vehicle.vehicle_id}"
            )

            max_orders = None

            if slot.shipper is not None:
                max_orders = (
                    slot.shipper.max_orders
                )

            workloads.append(
                AllocationWorkload(
                    workload_id=(
                        workload_id
                    ),
                    cluster_id=(
                        cluster_id
                    ),
                    depot_id=(
                        slot.vehicle.depot_id
                    ),
                    vehicle_id=(
                        slot.vehicle.vehicle_id
                    ),
                    shipper_id=(
                        None
                        if slot.shipper
                        is None
                        else slot.shipper
                        .shipper_id
                    ),
                    deliveries=list(
                        slot.deliveries
                    ),
                    total_weight_kg=float(
                        slot.total_weight_kg
                    ),
                    capacity_kg=float(
                        slot.vehicle.capacity_kg
                    ),
                    max_orders=(
                        max_orders
                    ),
                )
            )

        workloads.sort(
            key=lambda workload: (
                workload.cluster_id,
                workload.vehicle_id,
            )
        )

        return workloads

    def _cluster_total_weight(
        self,
        cluster: Cluster,
    ) -> float:
        return sum(
            self._delivery_weight_kg(
                delivery
            )
            for delivery
            in cluster.deliveries
        )

    def _delivery_weight_kg(
        self,
        delivery: Delivery,
    ) -> float:
        """
        Support the existing domain model
        without forcing a rename.

        Prefer weight_kg. Fall back to
        weight if the domain currently uses
        that attribute.
        """

        value = getattr(
            delivery,
            "weight_kg",
            None,
        )

        if value is None:
            value = getattr(
                delivery,
                "weight",
                None,
            )

        if value is None:
            raise ValueError(
                f"Delivery "
                f"{delivery.delivery_id} "
                "không có weight/weight_kg."
            )

        weight = float(
            value
        )

        if weight < 0.0:
            raise ValueError(
                f"Delivery "
                f"{delivery.delivery_id} "
                "có weight âm."
            )

        return weight

    def _validate_resources(
        self,
    ) -> None:
        vehicle_ids = [
            vehicle.vehicle_id
            for vehicle
            in self.vehicles
        ]

        if (
            len(
                vehicle_ids
            )
            != len(
                set(
                    vehicle_ids
                )
            )
        ):
            raise ValueError(
                "vehicle_id bị trùng."
            )

        shipper_ids = [
            shipper.shipper_id
            for shipper
            in self.shippers
        ]

        if (
            len(
                shipper_ids
            )
            != len(
                set(
                    shipper_ids
                )
            )
        ):
            raise ValueError(
                "shipper_id bị trùng."
            )

        for vehicle in (
            self.vehicles
        ):
            if (
                float(
                    vehicle.capacity_kg
                )
                <= 0.0
            ):
                raise ValueError(
                    f"Vehicle "
                    f"{vehicle.vehicle_id} "
                    "có capacity_kg <= 0."
                )

        for shipper in (
            self.shippers
        ):
            if (
                shipper.max_orders
                is not None
                and shipper.max_orders
                <= 0
            ):
                raise ValueError(
                    f"Shipper "
                    f"{shipper.shipper_id} "
                    "có max_orders <= 0."
                )