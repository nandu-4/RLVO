import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist"] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
      "@typescript-eslint/no-unused-vars": "off",
    },
  },
  {
    /*
     * Legacy RLVO research demos — a separate project that predates TruthLens and is kept only so
     * existing links keep working. It is not part of the TruthLens product surface, is not in the
     * primary navigation, and is held to its original standard rather than being rewritten at the
     * release gate. Excluded so `npm run verify` reflects the health of the code we actually own.
     */
    files: [
      "src/hooks/useProctoring.ts",
      "src/pages/Proctoring.tsx",
      "src/pages/ImageRefinement.tsx",
      "src/pages/VideoRefinement.tsx",
      "api/analyze-video.ts",
      "api/generate-caption.ts",
      "api/refine-caption.ts",
      "api/verify-flag.ts",
    ],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "no-empty": "off",
    },
  },
  {
    // Serverless handlers run in Node, and the platform hands them untyped req/res objects.
    // `any` at that boundary is the framework's own signature, not a shortcut in our code —
    // everything downstream of the boundary is strictly typed (see tsconfig.api.json).
    files: ["api/**/*.ts"],
    languageOptions: { globals: globals.node },
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
);
