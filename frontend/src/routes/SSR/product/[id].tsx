// src/routes/ssr/product/[id].tsx
import { createAsync, RouteDefinition } from "@solidjs/router";
import { Title, Meta } from "@solidjs/meta";

interface Product {
  id: string;
  name: string;
  price: number;
  description: string;
  image: string;
  inStock: boolean;
}

export const ssr = true;
// This runs on the server for each request
export const route = {
  load: ({ params }) => {
    return getProduct(params.id);
  },
} satisfies RouteDefinition;

async function getProduct(id: string): Promise<Product> {
  "user server";
  // Static dummy data instead of API call
  const dummyProducts: Record<string, Product> = {
    "123": {
      id: "123",
      name: "Premium Wireless Headphones",
      price: 199.99,
      description:
        "High-quality noise-cancelling headphones with 30-hour battery life, premium audio quality, and comfortable over-ear design. Perfect for music lovers and professionals alike.",
      image:
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=800&auto=format&fit=crop&q=60",
      inStock: true,
    },
    "456": {
      id: "456",
      name: "Smart Fitness Watch",
      price: 149.95,
      description:
        "Track your fitness goals with this advanced smartwatch featuring heart rate monitoring, GPS tracking, and a water-resistant design.",
      image:
        "https://images.unsplash.com/photo-1579586337278-3befd40fd17a?w=800&auto=format&fit=crop&q=60",
      inStock: true,
    },
    "789": {
      id: "789",
      name: "Professional Camera Lens",
      price: 599.0,
      description:
        "Professional-grade camera lens with superior optics, ideal for portrait and landscape photography.",
      image:
        "https://images.unsplash.com/photo-1617005082499-524276bd4d36?w=800&auto=format&fit=crop&q=60",
      inStock: false,
    },
  };

  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 300));

  // Check if product exists
  if (!dummyProducts[id]) {
    throw new Error(`Product ${id} not found`);
  }

  return dummyProducts[id];
}

export default function SSRProduct() {
  "user server";
  // You can change the ID to "123", "456", or "789" to see different products
  const productId = "123";
  const product = createAsync(() => getProduct(productId));

  return (
    <>
      <Title>{product()?.name || "Loading..."} - SSR Example</Title>
      <Meta
        name="description"
        content={product()?.description || "Product details"}
      />

      <div class="max-w-4xl mx-auto p-6">
        <div class="grid md:grid-cols-2 gap-8">
          <div>
            <img
              src={product()?.image || "/placeholder.jpg"}
              alt={product()?.name || "Product"}
              class="w-full h-96 object-cover rounded-lg"
            />
          </div>

          <div>
            <h1 class="text-3xl font-bold mb-4">{product()?.name}</h1>
            <p class="text-2xl text-green-600 font-semibold mb-4">
              ${product()?.price.toFixed(2)}
            </p>

            <div class="mb-4">
              <span
                class={`px-3 py-1 rounded-full text-sm ${
                  product()?.inStock
                    ? "bg-green-100 text-green-800"
                    : "bg-red-100 text-red-800"
                }`}
              >
                {product()?.inStock ? "In Stock" : "Out of Stock"}
              </span>
            </div>

            <p class="text-gray-700 mb-6">{product()?.description}</p>

            <button
              class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg disabled:opacity-50"
              disabled={!product()?.inStock}
            >
              Add to Cart
            </button>

            <div class="mt-6 text-sm text-gray-500">
              This page was server-rendered with fresh data on each request.
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
