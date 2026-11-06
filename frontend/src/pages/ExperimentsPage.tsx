import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  BarChart3,
  TrendingUp,
  ArrowRight,
} from "lucide-react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import Compare from "./Compare";
import { RLTrainingExplorer } from "@/components/research/RLTrainingExplorer";

export default function ExperimentsPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<"benchmark" | "training">("benchmark");

  return (
    <div className="space-y-4">
      {/* ============================================================== */}
      {/* 1. SCREEN HEADER (Clean, thesis-aligned, no badge clutter)      */}
      {/* ============================================================== */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-border/80 pb-3">
        <div>
          <h1 className="text-xl sm:text-2xl font-black tracking-tight text-foreground">
            Thực nghiệm và so sánh
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground mt-0.5">
            Đánh giá thuật toán định tuyến và kết quả huấn luyện học tăng cường.
          </p>
        </div>

        <Button
          size="sm"
          onClick={() => navigate("/toi-uu-rl")}
          className="gap-1.5 text-xs h-8 bg-teal-600 text-white hover:bg-teal-700 font-semibold shadow-xs shrink-0 cursor-pointer"
        >
          <span>Sang bàn lập tuyến</span>
          <ArrowRight className="size-3.5" />
        </Button>
      </div>

      {/* ============================================================== */}
      {/* 2. SUB-TABS: EXACTLY 2 MAIN TABS                               */}
      {/* ============================================================== */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="space-y-4">
        <TabsList className="bg-muted/50 p-1 border border-border/60 rounded-xl grid grid-cols-2 gap-1 w-full max-w-md">
          <TabsTrigger value="benchmark" className="gap-1.5 text-xs font-semibold data-[state=active]:bg-teal-600 data-[state=active]:text-white">
            <BarChart3 className="size-3.5" />
            <span>1. Kết quả thực nghiệm</span>
          </TabsTrigger>
          <TabsTrigger value="training" className="gap-1.5 text-xs font-semibold data-[state=active]:bg-teal-600 data-[state=active]:text-white">
            <TrendingUp className="size-3.5" />
            <span>2. Kết quả huấn luyện RL</span>
          </TabsTrigger>
        </TabsList>

        {/* TAB 1: KẾT QUẢ THỰC NGHIỆM */}
        <TabsContent value="benchmark" className="m-0 space-y-4">
          <Compare />
        </TabsContent>

        {/* TAB 2: HUẤN LUYỆN RL */}
        <TabsContent value="training" className="m-0 space-y-4">
          <RLTrainingExplorer />
        </TabsContent>
      </Tabs>
    </div>
  );
}
