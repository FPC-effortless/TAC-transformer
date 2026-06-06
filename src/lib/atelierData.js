import { useEffect, useMemo, useRef, useState } from "react";
import { ConvexReactClient } from "convex/react";
import { makeFunctionReference } from "convex/server";
import { INITIAL_CLIENTS, INITIAL_ORDERS } from "../utils/mockData";

const STORAGE_KEYS = {
  orders: "bibiere_orders",
  clients: "bibiere_clients"
};

const env = import.meta.env;

export const convexSettings = {
  url: env.VITE_CONVEX_URL || "",
  atelierQuery: env.VITE_CONVEX_ATELIER_QUERY || "",
  ordersQuery: env.VITE_CONVEX_ORDERS_QUERY || "",
  clientsQuery: env.VITE_CONVEX_CLIENTS_QUERY || "",
  upsertOrderMutation: env.VITE_CONVEX_UPSERT_ORDER_MUTATION || "",
  upsertClientMutation: env.VITE_CONVEX_UPSERT_CLIENT_MUTATION || "",
  saveSnapshotMutation: env.VITE_CONVEX_SAVE_SNAPSHOT_MUTATION || ""
};

export const isConvexConfigured = Boolean(
  convexSettings.url &&
    (convexSettings.atelierQuery ||
      (convexSettings.ordersQuery && convexSettings.clientsQuery))
);

export const convexClient = convexSettings.url
  ? new ConvexReactClient(convexSettings.url)
  : null;

const makeRef = (name) => (name ? makeFunctionReference(name) : null);

const refs = {
  atelierQuery: makeRef(convexSettings.atelierQuery),
  ordersQuery: makeRef(convexSettings.ordersQuery),
  clientsQuery: makeRef(convexSettings.clientsQuery),
  upsertOrderMutation: makeRef(convexSettings.upsertOrderMutation),
  upsertClientMutation: makeRef(convexSettings.upsertClientMutation),
  saveSnapshotMutation: makeRef(convexSettings.saveSnapshotMutation)
};

function readStoredJson(key, fallback) {
  try {
    const saved = localStorage.getItem(key);
    return saved ? JSON.parse(saved) : fallback;
  } catch {
    return fallback;
  }
}

function persistLocal(orders, clients) {
  localStorage.setItem(STORAGE_KEYS.orders, JSON.stringify(orders));
  localStorage.setItem(STORAGE_KEYS.clients, JSON.stringify(clients));
}

function normalizeRecord(record) {
  if (!record) return record;
  return {
    ...record,
    id: record.id || record._id
  };
}

function normalizePayload(payload) {
  if (!payload) return null;
  if (Array.isArray(payload)) return { orders: payload.map(normalizeRecord) };

  return {
    orders: Array.isArray(payload.orders) ? payload.orders.map(normalizeRecord) : undefined,
    clients: Array.isArray(payload.clients) ? payload.clients.map(normalizeRecord) : undefined
  };
}

function applyRemotePayload(payload, setOrders, setClients, setSyncStatus) {
  const normalized = normalizePayload(payload);
  if (!normalized) return;

  if (normalized.orders) {
    setOrders(normalized.orders);
  }
  if (normalized.clients) {
    setClients(normalized.clients);
  }

  setSyncStatus({
    mode: "convex",
    message: "Connected to Bibiere Convex",
    error: ""
  });
}

function subscribeToQuery(client, queryRef, onValue, onError) {
  if (!client || !queryRef) return () => {};

  const watch = client.watchQuery(queryRef, {});
  const read = () => {
    try {
      const result = watch.localQueryResult();
      if (result !== undefined) onValue(result);
    } catch (error) {
      onError(error);
    }
  };

  const unsubscribe = watch.onUpdate(read);
  read();
  return unsubscribe;
}

async function runMutation(client, mutationRef, args) {
  if (!client || !mutationRef) return;
  await client.mutation(mutationRef, args);
}

export function useAtelierData() {
  const [orders, setOrders] = useState(() =>
    readStoredJson(STORAGE_KEYS.orders, INITIAL_ORDERS)
  );
  const [clients, setClients] = useState(() =>
    readStoredJson(STORAGE_KEYS.clients, INITIAL_CLIENTS)
  );
  const [syncStatus, setSyncStatus] = useState(() => ({
    mode: isConvexConfigured ? "connecting" : "local",
    message: isConvexConfigured
      ? "Connecting to Bibiere Convex"
      : "Local workspace mode",
    error: ""
  }));
  const ordersRef = useRef(orders);
  const clientsRef = useRef(clients);

  useEffect(() => {
    ordersRef.current = orders;
    clientsRef.current = clients;
    persistLocal(orders, clients);
  }, [orders, clients]);

  useEffect(() => {
    if (!isConvexConfigured || !convexClient) return undefined;

    const handleError = (error) => {
      setSyncStatus({
        mode: "local",
        message: "Using local fallback",
        error: error?.message || "Convex sync failed"
      });
    };

    if (refs.atelierQuery) {
      return subscribeToQuery(
        convexClient,
        refs.atelierQuery,
        (payload) => applyRemotePayload(payload, setOrders, setClients, setSyncStatus),
        handleError
      );
    }

    const unsubscribeOrders = subscribeToQuery(
      convexClient,
      refs.ordersQuery,
      (payload) => applyRemotePayload({ orders: payload }, setOrders, setClients, setSyncStatus),
      handleError
    );
    const unsubscribeClients = subscribeToQuery(
      convexClient,
      refs.clientsQuery,
      (payload) => applyRemotePayload({ clients: payload }, setOrders, setClients, setSyncStatus),
      handleError
    );

    return () => {
      unsubscribeOrders();
      unsubscribeClients();
    };
  }, []);

  const api = useMemo(() => {
    const saveSnapshot = async (nextOrders, nextClients) => {
      try {
        await runMutation(convexClient, refs.saveSnapshotMutation, {
          orders: nextOrders,
          clients: nextClients
        });
      } catch (error) {
        setSyncStatus({
          mode: "local",
          message: "Saved locally, Convex write failed",
          error: error?.message || "Convex mutation failed"
        });
      }
    };

    return {
      orders,
      clients,
      syncStatus,
      saveOrders: async (nextOrders) => {
        ordersRef.current = nextOrders;
        setOrders(nextOrders);
        persistLocal(nextOrders, clientsRef.current);
        await saveSnapshot(nextOrders, clientsRef.current);
      },
      saveClients: async (nextClients) => {
        clientsRef.current = nextClients;
        setClients(nextClients);
        persistLocal(ordersRef.current, nextClients);
        await saveSnapshot(ordersRef.current, nextClients);
      },
      replaceData: async (nextOrders, nextClients) => {
        ordersRef.current = nextOrders;
        clientsRef.current = nextClients;
        setOrders(nextOrders);
        setClients(nextClients);
        persistLocal(nextOrders, nextClients);
        await saveSnapshot(nextOrders, nextClients);
      },
      upsertOrder: async (order) => {
        const activeOrders = ordersRef.current;
        const activeClients = clientsRef.current;
        const nextOrders = activeOrders.some((item) => item.id === order.id)
          ? activeOrders.map((item) => (item.id === order.id ? order : item))
          : [...activeOrders, order];

        ordersRef.current = nextOrders;
        setOrders(nextOrders);
        persistLocal(nextOrders, activeClients);

        try {
          await runMutation(convexClient, refs.upsertOrderMutation, { order });
          await saveSnapshot(nextOrders, activeClients);
        } catch (error) {
          setSyncStatus({
            mode: "local",
            message: "Order saved locally, Convex write failed",
            error: error?.message || "Convex mutation failed"
          });
        }
      },
      upsertClient: async (client) => {
        const activeOrders = ordersRef.current;
        const activeClients = clientsRef.current;
        const nextClients = activeClients.some((item) => item.id === client.id)
          ? activeClients.map((item) => (item.id === client.id ? client : item))
          : [...activeClients, client];

        clientsRef.current = nextClients;
        setClients(nextClients);
        persistLocal(activeOrders, nextClients);

        try {
          await runMutation(convexClient, refs.upsertClientMutation, { client });
          await saveSnapshot(activeOrders, nextClients);
        } catch (error) {
          setSyncStatus({
            mode: "local",
            message: "Client saved locally, Convex write failed",
            error: error?.message || "Convex mutation failed"
          });
        }
      }
    };
  }, [orders, clients, syncStatus]);

  return api;
}
