// src/routes/csr/dashboard.tsx
import { createSignal, createEffect, onMount } from "solid-js";
import { Title } from "@solidjs/meta";

interface User {
  id: number;
  name: string;
  email: string;
}

export default function CSRDashboard() {
  const [user, setUser] = createSignal<User | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);

  onMount(() => {
    // Using dummy data instead of API call
    setTimeout(() => {
      try {
        // Simulate a network delay
        const dummyUser = {
          id: 123,
          name: "John Doe",
          email: "john.doe@example.com",
        };

        setUser(dummyUser);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setLoading(false);
      }
    }, 1000); // Simulate 1 second loading delay
  });

  return (
    <>
      <Title>Dashboard - CSR Example</Title>
      <div class="p-6">
        <h1 class="text-2xl font-bold mb-4">User Dashboard (CSR)</h1>

        {loading() && (
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        )}

        {error() && (
          <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            Error: {error()}
          </div>
        )}

        {user() && (
          <div class="bg-white shadow rounded-lg p-6">
            <h2 class="text-xl font-semibold">Welcome, {user()!.name}!</h2>
            <p class="text-gray-600">Email: {user()!.email}</p>
            <div class="mt-4">
              <p class="text-sm text-gray-500">
                This content was loaded client-side after page load.
              </p>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
